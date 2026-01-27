#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // 基础检查
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());

    // 形状检查
    ASSERT(out->shape() == gate->shape() && out->shape() == up->shape(), 
           "SwigLU: out, gate, and up must have the same shape.");
    
    // 连续性检查
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), 
           "SwigLU: all tensors must be contiguous.");

    size_t num_elements = out->numel();

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), 
                           out->dtype(), num_elements);
    }

    EXCEPTION_UNSUPPORTED_DEVICE;
}
}