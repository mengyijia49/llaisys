#include "muxi_resource.hpp"

namespace llaisys::device::muxi {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_MUXI, device_id) {}
Resource::~Resource() = default;

} // namespace llaisys::device::muxi
