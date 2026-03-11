#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace {

float to_float(const std::byte *src, llaisysDataType_t dtype, size_t idx) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return reinterpret_cast<const float *>(src)[idx];
    case LLAISYS_DTYPE_F16:
        return llaisys::utils::cast<float>(reinterpret_cast<const llaisys::fp16_t *>(src)[idx]);
    case LLAISYS_DTYPE_BF16:
        return llaisys::utils::cast<float>(reinterpret_cast<const llaisys::bf16_t *>(src)[idx]);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

std::vector<float> copy_logits_to_host(llaisys::tensor_t logits) {
    const size_t n = logits->numel();
    std::vector<float> host_logits(n, 0.0f);

    const size_t bytes = n * logits->elementSize();
    std::vector<std::byte> raw(bytes);

    auto *runtime_api = llaisys::core::context().runtime().api();
    if (logits->deviceType() == LLAISYS_DEVICE_CPU) {
        std::copy_n(logits->data(), bytes, raw.data());
    } else {
        runtime_api->memcpy_sync(raw.data(), logits->data(), bytes, LLAISYS_MEMCPY_D2H);
    }

    for (size_t i = 0; i < n; ++i) {
        host_logits[i] = to_float(raw.data(), logits->dtype(), i);
    }
    return host_logits;
}

int64_t argmax_index(const std::vector<float> &logits) {
    return static_cast<int64_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

int64_t sample_from_logits(const std::vector<float> &logits, float temperature, int top_k, float top_p, uint64_t seed) {
    const size_t vocab = logits.size();
    if (vocab == 0) {
        return 0;
    }
    if (temperature <= 0.0f) {
        return argmax_index(logits);
    }

    int k = static_cast<int>(vocab);
    if (top_k > 0) {
        k = std::min<int>(top_k, static_cast<int>(vocab));
    }

    float p = top_p;
    if (p <= 0.0f || p > 1.0f) {
        p = 1.0f;
    }

    std::vector<size_t> candidate_ids(vocab);
    std::iota(candidate_ids.begin(), candidate_ids.end(), size_t{0});
    std::sort(candidate_ids.begin(), candidate_ids.end(), [&logits](size_t a, size_t b) { return logits[a] > logits[b]; });
    if (static_cast<size_t>(k) < candidate_ids.size()) {
        candidate_ids.resize(static_cast<size_t>(k));
    }

    const float safe_temperature = std::max(temperature, 1e-6f);

    std::vector<double> probs(candidate_ids.size(), 0.0);
    double max_scaled = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < candidate_ids.size(); ++i) {
        const double scaled = static_cast<double>(logits[candidate_ids[i]]) / static_cast<double>(safe_temperature);
        if (scaled > max_scaled) {
            max_scaled = scaled;
        }
        probs[i] = scaled;
    }

    double prob_sum = 0.0;
    for (double &v : probs) {
        v = std::exp(v - max_scaled);
        prob_sum += v;
    }

    if (prob_sum <= 0.0 || !std::isfinite(prob_sum)) {
        return static_cast<int64_t>(candidate_ids[0]);
    }

    if (p < 1.0f) {
        double cumulative = 0.0;
        size_t keep = 0;
        for (; keep < probs.size(); ++keep) {
            cumulative += probs[keep] / prob_sum;
            if (cumulative >= static_cast<double>(p)) {
                ++keep;
                break;
            }
        }
        if (keep == 0) {
            keep = 1;
        }
        if (keep < probs.size()) {
            candidate_ids.resize(keep);
            probs.resize(keep);
            prob_sum = std::accumulate(probs.begin(), probs.end(), 0.0);
        }
    }

    if (prob_sum <= 0.0 || !std::isfinite(prob_sum)) {
        return static_cast<int64_t>(candidate_ids[0]);
    }

    if (seed == 0) {
        std::random_device rd;
        seed = (static_cast<uint64_t>(rd()) << 32u) ^ static_cast<uint64_t>(rd());
    }

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, prob_sum);
    const double target = dist(rng);

    double acc = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        acc += probs[i];
        if (target <= acc) {
            return static_cast<int64_t>(candidate_ids[i]);
        }
    }

    return static_cast<int64_t>(candidate_ids.back());
}

} // namespace

namespace llaisys::ops {

void sample(tensor_t out_idx, tensor_t logits, float temperature, int top_k, float top_p, uint64_t seed) {
    CHECK_SAME_DEVICE(out_idx, logits);
    ASSERT(out_idx->dtype() == LLAISYS_DTYPE_I64, "Sample: output index tensor must use int64 dtype.");
    ASSERT(out_idx->numel() == 1, "Sample: output index tensor must contain exactly 1 element.");
    ASSERT(logits->isContiguous(), "Sample: logits tensor must be contiguous.");
    ASSERT(logits->ndim() == 1 || (logits->ndim() == 2 && logits->shape()[0] == 1), "Sample: logits shape must be [vocab] or [1, vocab].");
    ASSERT(logits->numel() > 0, "Sample: logits tensor must be non-empty.");

    ASSERT(
        logits->dtype() == LLAISYS_DTYPE_F32 || logits->dtype() == LLAISYS_DTYPE_F16 || logits->dtype() == LLAISYS_DTYPE_BF16,
        "Sample: logits dtype must be f32/f16/bf16.");

    llaisys::core::context().setDevice(logits->deviceType(), logits->deviceId());
    auto *runtime_api = llaisys::core::context().runtime().api();

    const std::vector<float> host_logits = copy_logits_to_host(logits);
    const int64_t sampled_idx = sample_from_logits(host_logits, temperature, top_k, top_p, seed);

    if (out_idx->deviceType() == LLAISYS_DEVICE_CPU) {
        reinterpret_cast<int64_t *>(out_idx->data())[0] = sampled_idx;
        return;
    }
    runtime_api->memcpy_sync(out_idx->data(), &sampled_idx, sizeof(int64_t), LLAISYS_MEMCPY_H2D);
}

} // namespace llaisys::ops

