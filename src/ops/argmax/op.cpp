#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/argmax_cpu.hpp" // 记得包含刚才写的新头文件

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    // 注意：根据任务要求，max_idx 和 max_val 是包含单元素的 1D 张量
    ASSERT(max_idx->numel() == 1 && max_val->numel() == 1, "Argmax: outputs must have 1 element.");
    ASSERT(vals->isContiguous(), "Argmax: input tensor must be contiguous.");

    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), 
                           vals->dtype(), vals->numel());
    }

    // 如果有其他设备，可以在这里继续 switch...
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops