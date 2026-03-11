from typing import Iterator, Literal, Sequence
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
                self._backend_kv = LIB_LLAISYS.llaisysQwen2KVCreat(self._backend_model, self._maxseq)

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
            from ctypes import c_size_t

            kv_cap = len(input_ids) + max_new_tokens
            if self._maxseq > 0:
                kv_cap = min(kv_cap, self._maxseq)
            kv_cap = max(kv_cap, 1)

            if self._backend_kv is not None:
                LIB_LLAISYS.llaisysQwen2KVDestroy(self._backend_kv)
                self._backend_kv = None
            self._backend_kv = LIB_LLAISYS.llaisysQwen2KVCreat(self._backend_model, c_size_t(kv_cap))

            use_sampled_infer = hasattr(LIB_LLAISYS, "llaisysQwen2ModelInferSampled")
            rng = random.Random(seed if seed is not None else None)

            def next_seed() -> int:
                if not use_sampled_infer:
                    return 0
                return rng.getrandbits(64)

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
            if self._backend_kv is not None:
                LIB_LLAISYS.llaisysQwen2KVDestroy(self._backend_kv)
                self._backend_kv = None
        except Exception:
            pass
        try:
            if self._backend_model is not None:
                LIB_LLAISYS.llaisysQwen2ModelDestroy(self._backend_model)
        except Exception:
            pass
