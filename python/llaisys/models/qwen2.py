from typing import Sequence
from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
from .. import Tensor
from ..libllaisys.qwen2 import LlaisysQwen2Meta
from pathlib import Path

import safetensors
import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        self._backend_model = None

        # attempt to create backend model
        try:
            # try to read model config if present to populate meta
            cfg = {}
            cfg_path = model_path / "config.json"
            if cfg_path.exists():
                import json

                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)

            meta = LlaisysQwen2Meta()
            meta.dtype = DataType.BF16
            # robustly extract fields from config
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

            dev = DeviceType.CPU if device == DeviceType.CPU else DeviceType.NVIDIA
            import ctypes

            self._backend_model = LIB_LLAISYS.llaisysQwen2ModelCreate(ctypes.byref(meta), dev, None, 0)

            for file in sorted(model_path.glob("*.safetensors")):
                data_ = safetensors.safe_open(file, framework="numpy", device="cpu")
                for name_ in data_.keys():
                    arr = data_.get_tensor(name_)
                    if not arr.flags["C_CONTIGUOUS"]:
                        arr = np.ascontiguousarray(arr)
                    if arr.dtype != np.float32:
                        arr = arr.astype(np.float32)

                    t = Tensor(shape=arr.shape, dtype=DataType.F32, device=DeviceType.CPU)
                    t.load(arr.ctypes.data)
                    LIB_LLAISYS.llaisysQwen2ModelSetWeight(self._backend_model, name_.encode("utf-8"), t.lib_tensor())

            LIB_LLAISYS.llaisysQwen2ModelFinalize(self._backend_model)

            # verify some required weights exist (best-effort)
            required = [b"model.norm.weight", b"embed_tokens.weight", b"lm_head.weight"]
            missing = []
            for r in required:
                try:
                    has = LIB_LLAISYS.llaisysQwen2ModelHasWeight(self._backend_model, r)
                except Exception:
                    has = 0
                if not has:
                    missing.append(r.decode("utf-8"))
            if missing:
                print("[llaisys qwen2] Warning: missing weights:", missing)
        except Exception as e:
            # backend unavailable or error during loading; fall back to HF
            self._backend_model = None

        if self._backend_model is None:
            if not HF_AVAILABLE:
                raise RuntimeError("Neither backend nor HuggingFace available for Qwen2 model")
            self.device = torch.device("cpu" if device == DeviceType.CPU else ("cuda" if torch.cuda.is_available() else "cpu"))
            self.model = AutoModelForCausalLM.from_pretrained(str(model_path), trust_remote_code=True, torch_dtype=torch.bfloat16)
            self.model.to(self.device)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if self._backend_model is not None:
            import ctypes
            from ctypes import c_int64, c_size_t

            arr = (c_int64 * len(inputs))(*inputs)
            out = LIB_LLAISYS.llaisysQwen2ModelInfer(self._backend_model, arr, c_size_t(len(inputs)))
            return [int(out)]

        input_ids = torch.tensor([list(inputs)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
        return outputs[0].tolist()

    def __del__(self):
        try:
            if self._backend_model is not None:
                LIB_LLAISYS.llaisysQwen2ModelDestroy(self._backend_model)
        except Exception:
            pass
