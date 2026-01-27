#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) CHECK_SAME_DEVICE(out, bias);

    // 形状校验
    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    size_t N = weight->shape()[0];
    
    ASSERT(weight->shape()[1] == K, "Linear: K dimension mismatch.");
    ASSERT(out->shape()[0] == M && out->shape()[1] == N, "Linear: output shape mismatch.");

    // 连续性校验
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), 
           "Linear: all main tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        // 注意：bias 可能为 nullptr，直接传入 data() 即可（底层做了 if(bias) 判断）
        const std::byte *bias_ptr = bias ? bias->data() : nullptr;
        return cpu::linear(out->data(), in->data(), weight->data(), bias_ptr,
                           out->dtype(), M, N, K);
    }

    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops