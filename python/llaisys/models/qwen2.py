from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterator, Literal, Optional, Sequence, Tuple
from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
from .. import Tensor
from ..libllaisys.qwen2 import LlaisysQwen2Meta
from pathlib import Path
import random

import safetensors

try:
    import torch
except Exception:
    torch = None

try:
    from transformers import AutoModelForCausalLM
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


@dataclass
class _PrefixCacheEntry:
    cache_key: str
    tokens: Tuple[int, ...]
    kv: int
    capacity: int


@dataclass
class _PendingPromptCache:
    cache_key: str
    prompt_tokens: Tuple[int, ...]
    kv: int
    capacity: int


class Qwen2:
    def __init__(
        self,
        model_path,
        device: DeviceType = DeviceType.CPU,
        backend: Literal["auto", "llaisys", "hf"] = "llaisys",
    ):
        backend = str(backend).lower()
        if backend not in ("auto", "llaisys", "hf"):
            raise ValueError("backend must be one of: auto, llaisys, hf")

        model_path = Path(model_path)
        self._backend_model = None
        self._backend_kv = None
        self._backend_preference = backend
        self._device = DeviceType.CPU if device == DeviceType.CPU else DeviceType.NVIDIA
        self._device_id = 0
        self._weight_tensors = []
        self._maxseq = 2048
        self._end_token = -1
        self._dtype = DataType.F32
        self._prefix_cache_limit = 24
        self._prefix_cache_pool: "OrderedDict[Tuple[str, Tuple[int, ...]], _PrefixCacheEntry]" = OrderedDict()
        self._pending_prompt_cache: Dict[str, _PendingPromptCache] = {}
        self._last_cache_info: Dict[str, int | bool | str] = {
            "cache_hit": False,
            "reused_prefix_tokens": 0,
        }

        try_backend = backend in ("auto", "llaisys")
        if try_backend:
            try:
                cfg = {}
                cfg_path = model_path / "config.json"
                if cfg_path.exists():
                    import json

                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)

                meta = LlaisysQwen2Meta()
                dtype_str = str(cfg.get("torch_dtype", "bfloat16")).lower()
                if "bfloat16" in dtype_str:
                    self._dtype = DataType.BF16
                elif "float16" in dtype_str or "half" in dtype_str:
                    self._dtype = DataType.F16
                else:
                    self._dtype = DataType.F32
                meta.dtype = self._dtype
                meta.nlayer = int(cfg.get("num_hidden_layers", cfg.get("n_layer", cfg.get("num_layers", 0))))
                meta.hs = int(cfg.get("hidden_size", cfg.get("d_model", 0)))
                meta.nh = int(cfg.get("num_attention_heads", cfg.get("n_head", 0)))
                meta.nkvh = int(cfg.get("num_key_value_heads", meta.nh if meta.nh else 0))
                if meta.nkvh == 0:
                    meta.nkvh = meta.nh
                meta.dh = int(cfg.get("head_dim", int(meta.hs / meta.nh) if meta.nh else 0))
                meta.di = int(cfg.get("intermediate_size", cfg.get("ffn_dim", meta.hs * 4 if meta.hs else 0)))
                meta.maxseq = int(cfg.get("max_position_embeddings", cfg.get("max_seq_len", 2048)))
                meta.voc = int(cfg.get("vocab_size", cfg.get("vocab_size", 0)))
                meta.epsilon = float(cfg.get("layer_norm_eps", cfg.get("eps", 1e-5)))
                meta.theta = float(cfg.get("rope_theta", cfg.get("theta", 10000.0)))
                meta.end_token = int(cfg.get("eos_token_id", cfg.get("end_token", -1)))
                self._maxseq = int(meta.maxseq)
                self._end_token = int(meta.end_token)

                import ctypes

                self._backend_model = LIB_LLAISYS.llaisysQwen2ModelCreate(ctypes.byref(meta), self._device, None, 0)

                if torch is None:
                    raise RuntimeError("PyTorch is required to load safetensors weights")

                if self._dtype == DataType.BF16:
                    target_torch_dtype = torch.bfloat16
                elif self._dtype == DataType.F16:
                    target_torch_dtype = torch.float16
                else:
                    target_torch_dtype = torch.float32

                for file in sorted(model_path.glob("*.safetensors")):
                    data_ = safetensors.safe_open(file, framework="pt", device="cpu")
                    for name_ in data_.keys():
                        ten = data_.get_tensor(name_).detach().cpu().contiguous()
                        if ten.dtype != target_torch_dtype:
                            ten = ten.to(target_torch_dtype)

                        t = Tensor(shape=ten.shape, dtype=self._dtype, device=self._device, device_id=self._device_id)
                        t.load(ten.data_ptr())
                        LIB_LLAISYS.llaisysQwen2ModelSetWeight(self._backend_model, name_.encode("utf-8"), t.lib_tensor())
                        self._weight_tensors.append(t)

                LIB_LLAISYS.llaisysQwen2ModelFinalize(self._backend_model)

                required_groups = [
                    [b"model.norm.weight", b"norm.weight"],
                    [b"model.embed_tokens.weight", b"embed_tokens.weight"],
                    [b"lm_head.weight"],
                ]
                missing = []
                for group in required_groups:
                    found = False
                    for r in group:
                        try:
                            has = LIB_LLAISYS.llaisysQwen2ModelHasWeight(self._backend_model, r)
                        except Exception:
                            has = 0
                        if has:
                            found = True
                            break
                    if not found:
                        missing.append(group[0].decode("utf-8"))
                if missing:
                    print("[llaisys qwen2] Warning: missing weights:", missing)
            except Exception as e:
                print(f"[llaisys qwen2] backend load failed: {e}")
                self._backend_model = None

        if self._backend_model is None:
            if backend == "llaisys":
                raise RuntimeError("Requested backend='llaisys', but backend model initialization failed")
            if not HF_AVAILABLE:
                raise RuntimeError("Neither backend nor HuggingFace available for Qwen2 model")
            if torch is None:
                raise RuntimeError("PyTorch is required for HuggingFace backend")
            use_cuda = (device != DeviceType.CPU) and torch.cuda.is_available()
            if device != DeviceType.CPU and not use_cuda:
                print("[llaisys qwen2] Warning: CUDA is unavailable, falling back to CPU for HF backend")
            self.device = torch.device("cuda" if use_cuda else "cpu")
            hf_dtype = torch.float16 if use_cuda else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(str(model_path), trust_remote_code=True, torch_dtype=hf_dtype)
            self.model.to(self.device)
            self.model.eval()

    def _destroy_kv(self, kv) -> None:
        if kv is None:
            return
        try:
            LIB_LLAISYS.llaisysQwen2KVDestroy(kv)
        except Exception:
            pass

    def _destroy_pending_prompt(self, cache_key: str) -> None:
        pending = self._pending_prompt_cache.pop(str(cache_key), None)
        if pending is not None:
            self._destroy_kv(pending.kv)

    def _backend_create_detached_kv(self, max_tokens: int):
        if not hasattr(LIB_LLAISYS, "llaisysQwen2KVCreatDetached"):
            return None
        return LIB_LLAISYS.llaisysQwen2KVCreatDetached(self._backend_model, max(1, int(max_tokens)))

    def _backend_clone_kv(self, kv, max_tokens: int):
        if kv is None or not hasattr(LIB_LLAISYS, "llaisysQwen2KVClone"):
            return None
        return LIB_LLAISYS.llaisysQwen2KVClone(kv, max(1, int(max_tokens)))

    def _replace_active_kv(self, kv) -> None:
        previous = self._backend_kv
        if hasattr(LIB_LLAISYS, "llaisysQwen2ModelSetKV"):
            rc = LIB_LLAISYS.llaisysQwen2ModelSetKV(self._backend_model, kv)
            if rc != 0:
                raise RuntimeError("failed to activate qwen2 KV cache")
        else:
            raise RuntimeError("llaisys shared library is missing llaisysQwen2ModelSetKV")
        self._backend_kv = kv
        if previous is not None and previous != kv:
            self._destroy_kv(previous)

    def _backend_prefill(self, token_ids: Sequence[int]) -> None:
        ids = [int(t) for t in token_ids]
        if not ids:
            return
        self._backend_infer(
            ids,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
            seed=0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            use_sampled_infer=False,
        )

    def _store_prefix_snapshot(self, cache_key: Optional[str], token_ids: Sequence[int], kv, capacity: int) -> None:
        if kv is None:
            return
        if not cache_key:
            self._destroy_kv(kv)
            return
        tokens = tuple(int(t) for t in token_ids)
        if not tokens:
            self._destroy_kv(kv)
            return
        namespaced = (str(cache_key), tokens)
        existing = self._prefix_cache_pool.pop(namespaced, None)
        if existing is not None:
            self._destroy_kv(existing.kv)
        self._prefix_cache_pool[namespaced] = _PrefixCacheEntry(
            cache_key=str(cache_key),
            tokens=tokens,
            kv=kv,
            capacity=max(1, int(capacity)),
        )
        self._prefix_cache_pool.move_to_end(namespaced)
        while len(self._prefix_cache_pool) > self._prefix_cache_limit:
            _, evicted = self._prefix_cache_pool.popitem(last=False)
            self._destroy_kv(evicted.kv)

    def _find_best_prefix_snapshot(self, cache_key: Optional[str], input_ids: Sequence[int]) -> Optional[_PrefixCacheEntry]:
        if not cache_key:
            return None
        wanted = [int(t) for t in input_ids]
        best = None
        for entry in reversed(self._prefix_cache_pool.values()):
            if entry.cache_key != str(cache_key):
                continue
            prefix_len = len(entry.tokens)
            if prefix_len == 0 or prefix_len >= len(wanted):
                continue
            if wanted[:prefix_len] != list(entry.tokens):
                continue
            best = entry
            break
        return best

    def _record_pending_prompt(self, cache_key: Optional[str], prompt_ids: Sequence[int], kv, capacity: int) -> None:
        if kv is None:
            return
        if not cache_key:
            self._destroy_kv(kv)
            return
        key = str(cache_key)
        self._destroy_pending_prompt(key)
        self._pending_prompt_cache[key] = _PendingPromptCache(
            cache_key=key,
            prompt_tokens=tuple(int(t) for t in prompt_ids),
            kv=kv,
            capacity=max(1, int(capacity)),
        )

    def get_last_cache_info(self) -> Dict[str, int | bool | str]:
        return dict(self._last_cache_info)

    def clear_prefix_cache(self, cache_key: Optional[str] = None) -> None:
        wanted = None if cache_key is None else str(cache_key)
        for key in list(self._prefix_cache_pool.keys()):
            namespaced, _ = key
            if wanted is not None and namespaced != wanted:
                continue
            entry = self._prefix_cache_pool.pop(key)
            self._destroy_kv(entry.kv)
        if wanted is None:
            for pending in list(self._pending_prompt_cache.values()):
                self._destroy_kv(pending.kv)
            self._pending_prompt_cache.clear()
            return
        self._destroy_pending_prompt(wanted)

    def commit_prefix_cache(
        self,
        cache_key: Optional[str],
        prompt_ids: Sequence[int],
        visible_completion_ids: Sequence[int],
        extra_capacity: int = 128,
    ) -> None:
        if self._backend_model is None or not cache_key:
            return
        key = str(cache_key)
        pending = self._pending_prompt_cache.pop(key, None)
        if pending is None:
            return

        prompt_tokens = tuple(int(t) for t in prompt_ids)
        if pending.prompt_tokens != prompt_tokens:
            self._destroy_kv(pending.kv)
            return

        visible_ids = [int(t) for t in visible_completion_ids]
        followup_prefix = list(prompt_tokens)
        if visible_ids:
            followup_prefix.extend(visible_ids[:-1])

        if not followup_prefix:
            self._destroy_kv(pending.kv)
            return

        target_capacity = max(int(pending.capacity), len(followup_prefix) + int(extra_capacity))
        scratch = self._backend_clone_kv(pending.kv, target_capacity)
        self._destroy_kv(pending.kv)
        if scratch is None:
            return

        self._replace_active_kv(scratch)
        if visible_ids[:-1]:
            self._backend_prefill(visible_ids[:-1])
        pool_snapshot = self._backend_clone_kv(self._backend_kv, target_capacity)
        self._store_prefix_snapshot(key, followup_prefix, pool_snapshot, target_capacity)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        seed: int = None,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 3,
        cache_key: Optional[str] = None,
    ):
        input_ids = [int(t) for t in inputs]
        if not input_ids:
            return []

        output_ids = list(input_ids)
        for token_id in self.stream_generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            seed=seed,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            cache_key=cache_key,
        ):
            output_ids.append(token_id)
        return output_ids

    def _backend_infer(
        self,
        token_ids: Sequence[int],
        top_k: int,
        top_p: float,
        temperature: float,
        seed: int,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        use_sampled_infer: bool,
    ) -> int:
        from ctypes import c_int, c_float, c_int64, c_size_t, c_uint64

        ids = [int(t) for t in token_ids]
        arr = (c_int64 * len(ids))(*ids)
        if use_sampled_infer:
            return int(
                LIB_LLAISYS.llaisysQwen2ModelInferSampled(
                    self._backend_model,
                    arr,
                    c_size_t(len(ids)),
                    c_int(int(top_k)),
                    c_float(float(top_p)),
                    c_float(float(temperature)),
                    c_uint64(int(seed) & ((1 << 64) - 1)),
                    c_float(float(repetition_penalty)),
                    c_int(int(no_repeat_ngram_size)),
                )
            )
        return int(LIB_LLAISYS.llaisysQwen2ModelInfer(self._backend_model, arr, c_size_t(len(ids))))

    def stream_generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        seed: int = None,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 3,
        cache_key: Optional[str] = None,
    ) -> Iterator[int]:
        input_ids = [int(t) for t in inputs]
        if not input_ids:
            return

        if max_new_tokens is None:
            max_new_tokens = 128
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens <= 0:
            return

        top_k = int(top_k)
        top_p = float(top_p)
        temperature = float(temperature)
        repetition_penalty = float(repetition_penalty)
        no_repeat_ngram_size = int(no_repeat_ngram_size)

        if self._backend_model is not None:
            supports_prefix_cache = (
                hasattr(LIB_LLAISYS, "llaisysQwen2KVCreatDetached")
                and hasattr(LIB_LLAISYS, "llaisysQwen2KVClone")
                and hasattr(LIB_LLAISYS, "llaisysQwen2ModelSetKV")
            )
            kv_cap = len(input_ids) + max_new_tokens
            if self._maxseq > 0:
                kv_cap = min(kv_cap, self._maxseq)
            kv_cap = max(kv_cap, 1)

            use_sampled_infer = hasattr(LIB_LLAISYS, "llaisysQwen2ModelInferSampled")
            rng = random.Random(seed if seed is not None else None)

            def next_seed() -> int:
                if not use_sampled_infer:
                    return 0
                return rng.getrandbits(64)

            if not supports_prefix_cache:
                self._last_cache_info = {
                    "cache_hit": False,
                    "reused_prefix_tokens": 0,
                    "cache_key": str(cache_key) if cache_key else "",
                }
                if self._backend_kv is not None:
                    LIB_LLAISYS.llaisysQwen2KVDestroy(self._backend_kv)
                    self._backend_kv = None
                self._backend_kv = LIB_LLAISYS.llaisysQwen2KVCreat(self._backend_model, kv_cap)
                next_token = self._backend_infer(
                    input_ids,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    seed=next_seed(),
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    use_sampled_infer=use_sampled_infer,
                )
                for _ in range(max_new_tokens):
                    if next_token is None:
                        break
                    next_token = int(next_token)
                    yield next_token
                    if self._end_token >= 0 and next_token == self._end_token:
                        break
                    next_token = self._backend_infer(
                        [next_token],
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        seed=next_seed(),
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        use_sampled_infer=use_sampled_infer,
                    )
                return

            prefix_entry = self._find_best_prefix_snapshot(cache_key, input_ids)
            prefix_len = len(prefix_entry.tokens) if prefix_entry is not None else 0
            self._last_cache_info = {
                "cache_hit": bool(prefix_len),
                "reused_prefix_tokens": int(prefix_len),
                "cache_key": str(cache_key) if cache_key else "",
            }

            active_kv = None
            if prefix_entry is not None:
                active_kv = self._backend_clone_kv(prefix_entry.kv, kv_cap)
            if active_kv is None:
                active_kv = self._backend_create_detached_kv(kv_cap)
            if active_kv is None:
                raise RuntimeError("failed to create qwen2 KV cache")

            self._replace_active_kv(active_kv)

            if len(input_ids) > 1 and prefix_len < len(input_ids) - 1:
                self._backend_prefill(input_ids[prefix_len:-1])

            if cache_key and len(input_ids) > 1:
                prompt_prefix_snapshot = self._backend_clone_kv(self._backend_kv, kv_cap)
                self._store_prefix_snapshot(cache_key, input_ids[:-1], prompt_prefix_snapshot, kv_cap)

            next_token = self._backend_infer(
                [input_ids[-1]],
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                seed=next_seed(),
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                use_sampled_infer=use_sampled_infer,
            )

            if cache_key:
                full_prompt_snapshot = self._backend_clone_kv(self._backend_kv, kv_cap)
                self._record_pending_prompt(cache_key, input_ids, full_prompt_snapshot, kv_cap)

            for _ in range(max_new_tokens):
                if next_token is None:
                    break
                next_token = int(next_token)
                yield next_token
                if self._end_token >= 0 and next_token == self._end_token:
                    break
                next_token = self._backend_infer(
                    [next_token],
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    seed=next_seed(),
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    use_sampled_infer=use_sampled_infer,
                )
            return

        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        do_sample = (top_k > 1) or (top_p < 0.999) or (abs(temperature - 1.0) > 1e-6)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": max(1.0, repetition_penalty),
            "no_repeat_ngram_size": max(0, no_repeat_ngram_size),
        }
        if do_sample:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "top_k": max(0, top_k),
                    "top_p": min(1.0, max(0.0, top_p)),
                    "temperature": max(1e-5, temperature),
                }
            )
        with torch.no_grad():
            outputs = self.model.generate(input_ids_tensor, **gen_kwargs)
        all_tokens = outputs[0].tolist()
        for token_id in all_tokens[len(input_ids):]:
            yield int(token_id)

    def __del__(self):
        try:
            self.clear_prefix_cache()
        except Exception:
            pass
        try:
            if self._backend_kv is not None:
                LIB_LLAISYS.llaisysQwen2KVDestroy(self._backend_kv)
                self._backend_kv = None
        except Exception:
            pass
        try:
            if self._backend_model is not None:
                self._backend_kv = None
                LIB_LLAISYS.llaisysQwen2ModelDestroy(self._backend_model)
        except Exception:
            pass
