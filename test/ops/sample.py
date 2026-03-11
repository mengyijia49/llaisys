import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import torch
import llaisys
from test_utils import llaisys_device, torch_device


def _tensor_from_torch(x: torch.Tensor, device_name: str) -> llaisys.Tensor:
    t = llaisys.Tensor(
        shape=tuple(x.shape),
        dtype=llaisys.DataType.F32 if x.dtype == torch.float32 else (
            llaisys.DataType.F16 if x.dtype == torch.float16 else llaisys.DataType.BF16
        ),
        device=llaisys_device(device_name),
        device_id=0,
    )
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    api.memcpy_sync(t.data_ptr(), x.data_ptr(), x.numel() * x.element_size(), llaisys.MemcpyKind.D2D)
    return t


def _read_scalar_i64(x: llaisys.Tensor, device_name: str) -> int:
    out = torch.zeros((1,), dtype=torch.int64, device=torch_device(device_name))
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    api.memcpy_sync(out.data_ptr(), x.data_ptr(), out.numel() * out.element_size(), llaisys.MemcpyKind.D2D)
    return int(out.item())


def test_topk1_equals_argmax(device_name: str):
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        logits = torch.randn((4096,), dtype=dtype, device=torch_device(device_name))
        logits_ll = _tensor_from_torch(logits, device_name)
        out_idx = llaisys.Tensor((1,), llaisys.DataType.I64, llaisys_device(device_name), 0)

        llaisys.Ops.sample(out_idx, logits_ll, temperature=0.7, top_k=1, top_p=0.1, seed=123)
        sampled = _read_scalar_i64(out_idx, device_name)
        expected = int(torch.argmax(logits.float()).item())
        assert sampled == expected, f"top_k=1 should equal argmax, got {sampled} expected {expected}"


def test_seed_reproducible(device_name: str):
    logits = torch.randn((2048,), dtype=torch.float32, device=torch_device(device_name))
    logits_ll = _tensor_from_torch(logits, device_name)
    out_a = llaisys.Tensor((1,), llaisys.DataType.I64, llaisys_device(device_name), 0)
    out_b = llaisys.Tensor((1,), llaisys.DataType.I64, llaisys_device(device_name), 0)

    llaisys.Ops.sample(out_a, logits_ll, temperature=0.8, top_k=40, top_p=0.9, seed=42)
    llaisys.Ops.sample(out_b, logits_ll, temperature=0.8, top_k=40, top_p=0.9, seed=42)

    a = _read_scalar_i64(out_a, device_name)
    b = _read_scalar_i64(out_b, device_name)
    assert a == b, "sampling with the same seed should be reproducible"


def test_topp_truncation(device_name: str):
    logits = torch.tensor([10.0, 9.0, 0.0, -2.0, -5.0], dtype=torch.float32, device=torch_device(device_name))
    logits_ll = _tensor_from_torch(logits, device_name)
    out_idx = llaisys.Tensor((1,), llaisys.DataType.I64, llaisys_device(device_name), 0)

    llaisys.Ops.sample(out_idx, logits_ll, temperature=1.0, top_k=0, top_p=0.3, seed=7)
    sampled = _read_scalar_i64(out_idx, device_name)
    assert sampled == 0, f"top_p truncation should keep the highest-probability token only, got {sampled}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    args = parser.parse_args()

    print(f"Testing Ops.sample on {args.device}")
    test_topk1_equals_argmax(args.device)
    test_seed_reproducible(args.device)
    test_topp_truncation(args.device)
    print("\033[92mTest passed!\033[0m\n")

