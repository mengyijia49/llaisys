#include "llaisys/models/qwen2.h"
#include "llaisys/tensor.h"
#include "llaisys_tensor.hpp"
#include <cstring>
#include <string>
#include <unordered_map>
#include <regex>
#include <vector>
#include <iostream>
#include "llaisys/ops.h"

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    // store arbitrary named tensors provided from python
    std::unordered_map<std::string, llaisysTensor_t> weight_map;
    // simple KV cache pointer (optional)
    // not owning tensors, just storing handles
    struct KVCache *kv_cache = nullptr;
};

extern "C" {

__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    if (!meta) return nullptr;
    LlaisysQwen2Model *m = new LlaisysQwen2Model();
    m->meta = *meta;
    memset(&m->weights, 0, sizeof(m->weights));
    return m;
}

__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
    if (!model) return;
    model->weight_map.clear();
    if (model->weights.attn_norm_w) delete[] model->weights.attn_norm_w;
    if (model->weights.attn_q_w) delete[] model->weights.attn_q_w;
    if (model->weights.attn_q_b) delete[] model->weights.attn_q_b;
    if (model->weights.attn_k_w) delete[] model->weights.attn_k_w;
    if (model->weights.attn_k_b) delete[] model->weights.attn_k_b;
    if (model->weights.attn_v_w) delete[] model->weights.attn_v_w;
    if (model->weights.attn_v_b) delete[] model->weights.attn_v_b;
    if (model->weights.attn_o_w) delete[] model->weights.attn_o_w;
    if (model->weights.mlp_norm_w) delete[] model->weights.mlp_norm_w;
    if (model->weights.mlp_gate_w) delete[] model->weights.mlp_gate_w;
    if (model->weights.mlp_up_w) delete[] model->weights.mlp_up_w;
    if (model->weights.mlp_down_w) delete[] model->weights.mlp_down_w;
    delete model;
}

__export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
    if (!model) return nullptr;
    return &model->weights;
}

static bool str_contains_any(const std::string &s, const std::vector<std::string> &subs) {
    for (auto &p : subs) if (s.find(p) != std::string::npos) return true;
    return false;
}

static bool tensor_matches_shape_and_dtype(llaisysTensor_t t, const std::vector<size_t> &expected_shape, llaisysDataType_t dtype) {
    if (!t) return false;
    try {
        auto &s = t->tensor->shape();
        if (s.size() != expected_shape.size()) return false;
        for (size_t i = 0; i < s.size(); ++i) if (s[i] != expected_shape[i]) return false;
        if (t->tensor->dtype() != dtype) return false;
        return true;
    } catch (...) {
        return false;
    }
}

__export int llaisysQwen2ModelSetWeight(struct LlaisysQwen2Model * model, const char * name, llaisysTensor_t tensor) {
    if (!model || !name) return -1;
    std::string sname(name);
    model->weight_map[sname] = tensor;

    // top-level mapping heuristics
    if (str_contains_any(sname, {"embed_tokens", "word_embeddings", "tok_embeddings", "token_embedding", "embed."})) {
        std::vector<size_t> exp = {model->meta.voc, model->meta.hs};
        if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.in_embed = tensor;
        else std::cerr << "[llaisys qwen2] Warning: in_embed shape/dtype mismatch for " << sname << std::endl;
        return 0;
    }
    if (str_contains_any(sname, {"lm_head", "output_projection", "out_proj", "out_embed", "head"})) {
        std::vector<size_t> exp = {model->meta.voc, model->meta.hs};
        if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.out_embed = tensor;
        else std::cerr << "[llaisys qwen2] Warning: out_embed shape/dtype mismatch for " << sname << std::endl;
        return 0;
    }
    if (str_contains_any(sname, {"model.norm.weight", "ln_f.weight", "final_layernorm.weight", "out_norm.weight", "norm.weight"})) {
        // avoid matching layer-norm inside layers by checking common patterns
        std::vector<size_t> exp = {model->meta.hs};
        if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.out_norm_w = tensor;
        else std::cerr << "[llaisys qwen2] Warning: out_norm_w shape/dtype mismatch for " << sname << std::endl;
        return 0;
    }

    // per-layer: try to find layer index using several common patterns
    std::smatch m;
    std::vector<std::regex> layer_patterns = {
        std::regex("layers\\.(\\d+)"),
        std::regex("blocks\\.(\\d+)"),
        std::regex("h\\.(\\d+)")
    };
    int idx = -1;
    for (auto &rp : layer_patterns) {
        if (std::regex_search(sname, m, rp)) { idx = std::stoi(m[1].str()); break; }
    }
    if (idx >= 0 && (size_t)idx < model->meta.nlayer) {
        // per-layer mapping with shape/dtype validation
        size_t hs = model->meta.hs;
        size_t nh = model->meta.nh;
        size_t nkv = model->meta.nkvh;
        size_t dh = model->meta.dh;
        size_t di = model->meta.di;

        // input layer norm weight [hs]
        if (str_contains_any(sname, {"input_layernorm.weight", "ln_1.weight", "layernorm_before.weight", "attention_layernorm.weight"})) {
            std::vector<size_t> exp = {hs};
            if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.attn_norm_w[idx] = tensor;
            else std::cerr << "[llaisys qwen2] Warning: attn_norm_w["<<idx<<"] shape/dtype mismatch" << std::endl;
            return 0;
        }

        // q projection [nh*dh, hs], bias [nh*dh]
        if (str_contains_any(sname, {"q_proj.weight", "q.weight", "qkv.q.weight", "wq.weight", "attn.q.weight"})) {
            std::vector<size_t> exp = {nh * dh, hs};
            if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.attn_q_w[idx] = tensor;
            else std::cerr << "[llaisys qwen2] Warning: attn_q_w["<<idx<<"] shape/dtype mismatch" << std::endl;
            return 0;
        }
        if (str_contains_any(sname, {"q_proj.bias", "q.bias", "wq.bias"})) {
            std::vector<size_t> exp = {nh * dh};
            if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.attn_q_b[idx] = tensor;
            else std::cerr << "[llaisys qwen2] Warning: attn_q_b["<<idx<<"] shape/dtype mismatch" << std::endl;
            return 0;
        }

        // k projection [nkv*dh, hs], bias [nkv*dh]
        if (str_contains_any(sname, {"k_proj.weight", "k.weight", "wk.weight", "attn.k.weight"})) {
            std::vector<size_t> exp = {nkv * dh, hs};
            if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.attn_k_w[idx] = tensor;
            else std::cerr << "[llaisys qwen2] Warning: attn_k_w["<<idx<<"] shape/dtype mismatch" << std::endl;
            return 0;
        }
        if (str_contains_any(sname, {"k_proj.bias", "k.bias", "wk.bias"})) {
            std::vector<size_t> exp = {nkv * dh};
            if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.attn_k_b[idx] = tensor;
            else std::cerr << "[llaisys qwen2] Warning: attn_k_b["<<idx<<"] shape/dtype mismatch" << std::endl;
            return 0;
        }

        // v projection [nkv*dh, hs], bias [nkv*dh]
        if (str_contains_any(sname, {"v_proj.weight", "v.weight", "wv.weight", "attn.v.weight"})) {
            std::vector<size_t> exp = {nkv * dh, hs};
            if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.attn_v_w[idx] = tensor;
            else std::cerr << "[llaisys qwen2] Warning: attn_v_w["<<idx<<"] shape/dtype mismatch" << std::endl;
            return 0;
        }
        if (str_contains_any(sname, {"v_proj.bias", "v.bias", "wv.bias"})) {
            std::vector<size_t> exp = {nkv * dh};
            if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.attn_v_b[idx] = tensor;
            else std::cerr << "[llaisys qwen2] Warning: attn_v_b["<<idx<<"] shape/dtype mismatch" << std::endl;
            return 0;
        }

        // out projection (attention) [hs, nh*dh]
        if (str_contains_any(sname, {"o_proj.weight", "o.weight", "wo.weight", "attn.o.weight"})) {
            std::vector<size_t> exp = {hs, nh * dh};
            if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.attn_o_w[idx] = tensor;
            else std::cerr << "[llaisys qwen2] Warning: attn_o_w["<<idx<<"] shape/dtype mismatch" << std::endl;
            return 0;
        }

        // post-attention norm
        if (str_contains_any(sname, {"post_attention_layernorm.weight", "ln_2.weight", "layernorm_after.weight"})) {
            std::vector<size_t> exp = {hs};
            if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.mlp_norm_w[idx] = tensor;
            else std::cerr << "[llaisys qwen2] Warning: mlp_norm_w["<<idx<<"] shape/dtype mismatch" << std::endl;
            return 0;
        }

        // MLP gate/up/down
        if (str_contains_any(sname, {"mlp.gate_proj.weight", "gate_proj.weight", "dense_h_to_4h.weight", "gate.weight"})) {
            std::vector<size_t> exp = {di, hs};
            if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.mlp_gate_w[idx] = tensor;
            else std::cerr << "[llaisys qwen2] Warning: mlp_gate_w["<<idx<<"] shape/dtype mismatch" << std::endl;
            return 0;
        }
        if (str_contains_any(sname, {"mlp.up_proj.weight", "up_proj.weight", "up.weight"})) {
            std::vector<size_t> exp = {di, hs};
            if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.mlp_up_w[idx] = tensor;
            else std::cerr << "[llaisys qwen2] Warning: mlp_up_w["<<idx<<"] shape/dtype mismatch" << std::endl;
            return 0;
        }
        if (str_contains_any(sname, {"mlp.down_proj.weight", "down_proj.weight", "down.weight"})) {
            std::vector<size_t> exp = {hs, di};
            if (tensor_matches_shape_and_dtype(tensor, exp, model->meta.dtype)) model->weights.mlp_down_w[idx] = tensor;
            else std::cerr << "[llaisys qwen2] Warning: mlp_down_w["<<idx<<"] shape/dtype mismatch" << std::endl;
            return 0;
        }
    }

    return 0;
}

__export int llaisysQwen2ModelFinalize(struct LlaisysQwen2Model * model) {
    if (!model) return -1;
    size_t n = model->meta.nlayer;
    if (n > 0) {
        if (!model->weights.attn_norm_w) model->weights.attn_norm_w = new llaisysTensor_t[n]();
        if (!model->weights.attn_q_w) model->weights.attn_q_w = new llaisysTensor_t[n]();
        if (!model->weights.attn_q_b) model->weights.attn_q_b = new llaisysTensor_t[n]();
        if (!model->weights.attn_k_w) model->weights.attn_k_w = new llaisysTensor_t[n]();
        if (!model->weights.attn_k_b) model->weights.attn_k_b = new llaisysTensor_t[n]();
        if (!model->weights.attn_v_w) model->weights.attn_v_w = new llaisysTensor_t[n]();
        if (!model->weights.attn_v_b) model->weights.attn_v_b = new llaisysTensor_t[n]();
        if (!model->weights.attn_o_w) model->weights.attn_o_w = new llaisysTensor_t[n]();
        if (!model->weights.mlp_norm_w) model->weights.mlp_norm_w = new llaisysTensor_t[n]();
        if (!model->weights.mlp_gate_w) model->weights.mlp_gate_w = new llaisysTensor_t[n]();
        if (!model->weights.mlp_up_w) model->weights.mlp_up_w = new llaisysTensor_t[n]();
        if (!model->weights.mlp_down_w) model->weights.mlp_down_w = new llaisysTensor_t[n]();
    }

    // quick validation: warn about missing critical weights
    std::vector<std::string> missing;
    if (!model->weights.in_embed) missing.push_back("in_embed");
    if (!model->weights.out_norm_w) missing.push_back("out_norm_w");
    if (!model->weights.out_embed) missing.push_back("out_embed");
    if (n > 0) {
        for (size_t i = 0; i < n; ++i) {
            size_t hs = model->meta.hs;
            size_t nh = model->meta.nh;
            size_t nkv = model->meta.nkvh;
            size_t dh = model->meta.dh;
            size_t di = model->meta.di;

            std::vector<size_t> qw = {nh * dh, hs};
            std::vector<size_t> kw = {nkv * dh, hs};
            std::vector<size_t> vw = {nkv * dh, hs};
            std::vector<size_t> ow = {hs, nh * dh};
            std::vector<size_t> normv = {hs};
            std::vector<size_t> gatew = {di, hs};
            std::vector<size_t> downw = {hs, di};

            if (!model->weights.attn_q_w[i] || !tensor_matches_shape_and_dtype(model->weights.attn_q_w[i], qw, model->meta.dtype)) missing.push_back("attn_q_w[" + std::to_string(i) + "]");
            if (!model->weights.attn_k_w[i] || !tensor_matches_shape_and_dtype(model->weights.attn_k_w[i], kw, model->meta.dtype)) missing.push_back("attn_k_w[" + std::to_string(i) + "]");
            if (!model->weights.attn_v_w[i] || !tensor_matches_shape_and_dtype(model->weights.attn_v_w[i], vw, model->meta.dtype)) missing.push_back("attn_v_w[" + std::to_string(i) + "]");
            if (!model->weights.attn_o_w[i] || !tensor_matches_shape_and_dtype(model->weights.attn_o_w[i], ow, model->meta.dtype)) missing.push_back("attn_o_w[" + std::to_string(i) + "]");
            if (!model->weights.mlp_norm_w[i] || !tensor_matches_shape_and_dtype(model->weights.mlp_norm_w[i], normv, model->meta.dtype)) missing.push_back("mlp_norm_w[" + std::to_string(i) + "]");
            if (!model->weights.mlp_gate_w[i] || !tensor_matches_shape_and_dtype(model->weights.mlp_gate_w[i], gatew, model->meta.dtype)) missing.push_back("mlp_gate_w[" + std::to_string(i) + "]");
            if (!model->weights.mlp_down_w[i] || !tensor_matches_shape_and_dtype(model->weights.mlp_down_w[i], downw, model->meta.dtype)) missing.push_back("mlp_down_w[" + std::to_string(i) + "]");
        }
    }
    if (!missing.empty()) {
        std::cerr << "[llaisys qwen2] Warning: missing weights:";
        for (auto &s : missing) std::cerr << " " << s;
        std::cerr << std::endl;
    }

    return 0;
}

// Simple KV cache structure and APIs
struct KVCache {
    size_t max_tokens;
    std::vector<llaisysTensor_t> keys;
    std::vector<llaisysTensor_t> vals;
};

__export void *llaisysQwen2KVCreat(struct LlaisysQwen2Model * model, size_t max_tokens) {
    KVCache *kv = new KVCache();
    kv->max_tokens = max_tokens;
    kv->keys.reserve(max_tokens);
    kv->vals.reserve(max_tokens);
    if (model) model->kv_cache = kv;
    return (void *)kv;
}

__export void llaisysQwen2KVDestroy(void *kv) {
    if (!kv) return;
    KVCache *c = (KVCache *)kv;
    c->keys.clear();
    c->vals.clear();
    delete c;
}

__export int llaisysQwen2KVAppend(void *kv, llaisysTensor_t k, llaisysTensor_t v) {
    if (!kv) return -1;
    KVCache *c = (KVCache *)kv;
    if (c->keys.size() >= c->max_tokens) return -1;
    c->keys.push_back(k);
    c->vals.push_back(v);
    return 0;
}

__export size_t llaisysQwen2KVLen(void *kv) {
    if (!kv) return 0;
    KVCache *c = (KVCache *)kv;
    return c->keys.size();
}

__export uint8_t llaisysQwen2ModelHasWeight(struct LlaisysQwen2Model * model, const char * name) {
    if (!model || !name) return 0;
    std::string sname(name);
    auto it = model->weight_map.find(sname);
    return it != model->weight_map.end() ? 1 : 0;
}

__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
    // More complete per-layer feedforward inference (attention skipped):
    if (!model) return -1;
    std::cerr << "[llaisys qwen2] infer entry" << std::endl;
    if (ntoken == 0 || token_ids == nullptr) return model->meta.end_token;

    int64_t last = token_ids[ntoken - 1];

    // require embedding and out projection
    if (!model->weights.in_embed || !model->weights.out_embed) {
        return last;
    }

    using namespace llaisys;

    // create index tensor [1]
    std::vector<size_t> idx_shape = {1};
    auto idx_tensor = Tensor::create(idx_shape, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
    LlaisysTensor *idx = new LlaisysTensor{idx_tensor};
    idx->tensor->load(&last);

    // embedding -> x [1, hs]
    std::vector<size_t> emb_shape = {1, model->meta.hs};
    auto x_tensor = Tensor::create(emb_shape, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
    LlaisysTensor *x = new LlaisysTensor{x_tensor};
    llaisysEmbedding(x, idx, model->weights.in_embed);
    std::cerr << "[llaisys qwen2] after embedding" << std::endl;

    // per-layer processing
    size_t n = model->meta.nlayer;
    for (size_t i = 0; i < n; ++i) {
        std::cerr << "[llaisys qwen2] layer " << i << " start" << std::endl;
        // optional rms_norm -> normed [1, hs]
        LlaisysTensor *norm = nullptr;
        if (model->weights.attn_norm_w && model->weights.attn_norm_w[i]) {
            auto norm_tensor = Tensor::create(emb_shape, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
            norm = new LlaisysTensor{norm_tensor};
            llaisysRmsNorm(norm, x, model->weights.attn_norm_w[i], model->meta.epsilon);
        }

        // Attention: compute q,k,v, append to KV and run self-attention
        bool has_attn = model->weights.attn_norm_w && model->weights.attn_q_w && model->weights.attn_k_w && model->weights.attn_v_w && model->weights.attn_o_w &&
                        model->weights.attn_norm_w[i] && model->weights.attn_q_w[i] && model->weights.attn_k_w[i] && model->weights.attn_v_w[i] && model->weights.attn_o_w[i];

        if (has_attn) {
            // q: project norm -> [1, nh*dh] then view [1, nh, dh]
            size_t nh = model->meta.nh;
            size_t dh = model->meta.dh;
            size_t nkv = model->meta.nkvh;

            std::vector<size_t> qflat_shape = {1, nh * dh};
            auto qflat_tensor = Tensor::create(qflat_shape, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
            LlaisysTensor *qflat = new LlaisysTensor{qflat_tensor};
            llaisysLinear(qflat, norm ? norm : x, model->weights.attn_q_w[i], model->weights.attn_q_b && model->weights.attn_q_b[i] ? model->weights.attn_q_b[i] : nullptr);
            // view to [1, nh, dh]
            LlaisysTensor *q = nullptr;
            try {
                auto q_view = qflat->tensor->view({1, nh, dh});
                q = new LlaisysTensor{q_view};
            } catch (...) {
                delete qflat; qflat = nullptr;
            }

            // k: project norm -> [1, nkv*dh] then view [1, nkv, dh]
            std::vector<size_t> kflat_shape = {1, nkv * dh};
            auto kflat_tensor = Tensor::create(kflat_shape, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
            LlaisysTensor *kflat = new LlaisysTensor{kflat_tensor};
            llaisysLinear(kflat, norm ? norm : x, model->weights.attn_k_w[i], model->weights.attn_k_b && model->weights.attn_k_b[i] ? model->weights.attn_k_b[i] : nullptr);
            LlaisysTensor *k = nullptr;
            try {
                auto k_view = kflat->tensor->view({1, nkv, dh});
                k = new LlaisysTensor{k_view};
            } catch (...) {
                delete kflat; kflat = nullptr;
            }

            // v: project norm -> [1, nkv*dh] then view [1, nkv, dh]
            std::vector<size_t> vflat_shape = {1, nkv * dh};
            auto vflat_tensor = Tensor::create(vflat_shape, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
            LlaisysTensor *vflat = new LlaisysTensor{vflat_tensor};
            llaisysLinear(vflat, norm ? norm : x, model->weights.attn_v_w[i], model->weights.attn_v_b && model->weights.attn_v_b[i] ? model->weights.attn_v_b[i] : nullptr);
            LlaisysTensor *v = nullptr;
            try {
                auto v_view = vflat->tensor->view({1, nkv, dh});
                v = new LlaisysTensor{v_view};
            } catch (...) {
                delete vflat; vflat = nullptr;
            }

            // append k/v to kv cache if exists (or create ephemeral vectors)
            KVCache *kv = model->kv_cache;
            if (kv) {
                // note: KVAppend stores handles, we transfer ownership semantics to KVCache (do not delete appended tensors here)
                llaisysQwen2KVAppend(kv, k, v);
            }

            // build k_all and v_all from KV entries
            size_t total_len = 1;
            std::vector<LlaisysTensor *> kv_keys;
            std::vector<LlaisysTensor *> kv_vals;
            if (kv && !kv->keys.empty()) {
                total_len = kv->keys.size();
                kv_keys = kv->keys;
                kv_vals = kv->vals;
            } else {
                total_len = 1;
                kv_keys = {k};
                kv_vals = {v};
            }

            // create k_all [total_len, nkv, dh], v_all [total_len, nkv, dh]
            std::vector<size_t> k_all_shape = {total_len, nkv, dh};
            auto k_all_t = Tensor::create(k_all_shape, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
            auto v_all_t = Tensor::create(k_all_shape, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
            LlaisysTensor *k_all = new LlaisysTensor{k_all_t};
            LlaisysTensor *v_all = new LlaisysTensor{v_all_t};

            // copy each stored key/val into k_all/v_all
            for (size_t j = 0; j < total_len; ++j) {
                // compute byte size per entry
                size_t elems = nkv * dh;
                size_t bytes = elems * model->weights.in_embed->tensor->elementSize();
                // source pointers
                const std::byte *src_k = kv_keys[j]->tensor->data();
                const std::byte *src_v = kv_vals[j]->tensor->data();
                std::byte *dst_k = k_all->tensor->data() + j * elems * model->weights.in_embed->tensor->elementSize();
                std::byte *dst_v = v_all->tensor->data() + j * elems * model->weights.in_embed->tensor->elementSize();
                // memcpy (works for CPU)
                std::memcpy(dst_k, src_k, bytes);
                std::memcpy(dst_v, src_v, bytes);
            }

            // prepare attn_out [1, nh, dh]
            std::vector<size_t> attn_shape = {1, nh, dh};
            auto attn_t = Tensor::create(attn_shape, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
            LlaisysTensor *attn_out = new LlaisysTensor{attn_t};

            float scale = 1.0f / std::sqrt((float)dh);
            llaisysSelfAttention(attn_out, q, k_all, v_all, scale);

            // flatten attn_out to [1, hs] and add to x
            try {
                auto attn_flat = attn_out->tensor->view({1, model->meta.hs});
                LlaisysTensor *attn_flat_t = new LlaisysTensor{attn_flat};
                auto new_x_tensor = Tensor::create(emb_shape, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
                LlaisysTensor *new_x = new LlaisysTensor{new_x_tensor};
                llaisysAdd(new_x, x, attn_flat_t);
                delete x; x = new_x;
                delete attn_flat_t;
            } catch (...) {
                // ignore
            }

            // cleanup temporaries we own (do not delete kv-stored tensors)
            delete qflat; if (q) delete q;
            if (!kv) { delete kflat; if (k) delete k; delete vflat; if (v) delete v; }
            delete k_all; delete v_all; delete attn_out;
        }

        // MLP: only run if the expected mlp weights are present
        bool has_mlp = model->weights.mlp_norm_w && model->weights.mlp_gate_w && model->weights.mlp_up_w && model->weights.mlp_down_w &&
                       model->weights.mlp_norm_w[i] && model->weights.mlp_gate_w[i] && model->weights.mlp_up_w[i] && model->weights.mlp_down_w[i];

        if (has_mlp) {
            std::cerr << "[llaisys qwen2] layer " << i << " mlp present" << std::endl;
            auto mlp_in = Tensor::create(emb_shape, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
            LlaisysTensor *mlp_in_t = new LlaisysTensor{mlp_in};
            llaisysRmsNorm(mlp_in_t, x, model->weights.mlp_norm_w[i], model->meta.epsilon);

            // gate [1, di]
            std::vector<size_t> gate_shape = {1, model->meta.di};
            auto gate_tensor = Tensor::create(gate_shape, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
            LlaisysTensor *gate = new LlaisysTensor{gate_tensor};
            // up [1, di]
            auto up_tensor = Tensor::create(gate_shape, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
            LlaisysTensor *up = new LlaisysTensor{up_tensor};

            // linear projections
            llaisysLinear(gate, mlp_in_t, model->weights.mlp_gate_w[i], nullptr);
            llaisysLinear(up, mlp_in_t, model->weights.mlp_up_w[i], nullptr);

            // swiglu -> act [1, di]
            auto act_tensor = Tensor::create(gate_shape, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
            LlaisysTensor *act = new LlaisysTensor{act_tensor};
            llaisysSwiGLU(act, gate, up);

            // down projection -> out [1, hs]
            auto down_tensor = Tensor::create(emb_shape, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
            LlaisysTensor *down = new LlaisysTensor{down_tensor};
            llaisysLinear(down, act, model->weights.mlp_down_w[i], nullptr);

            // residual: x = x + down
            auto new_x_tensor = Tensor::create(emb_shape, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
            LlaisysTensor *new_x = new LlaisysTensor{new_x_tensor};
            llaisysAdd(new_x, x, down);

            // swap x
            delete x; x = new_x;

            // cleanup temporaries
            delete mlp_in_t;
            delete gate;
            delete up;
            delete act;
            delete down;
            std::cerr << "[llaisys qwen2] layer " << i << " mlp done" << std::endl;
        }

        if (norm) delete norm;
        std::cerr << "[llaisys qwen2] layer " << i << " end" << std::endl;
    }

    // logits
    std::cerr << "[llaisys qwen2] before logits" << std::endl;
    std::cerr << "[llaisys qwen2] x ptr=" << x << " out_embed ptr=" << model->weights.out_embed << std::endl;
    std::vector<size_t> logits_shape = {1, model->meta.voc};
    auto logits_tensor = Tensor::create(logits_shape, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
    LlaisysTensor *logits = new LlaisysTensor{logits_tensor};
    std::cerr << "[llaisys qwen2] calling llaisysLinear for logits" << std::endl;
    llaisysLinear(logits, x, model->weights.out_embed, nullptr);
    std::cerr << "[llaisys qwen2] after llaisysLinear for logits" << std::endl;

    // argmax
    std::vector<size_t> one_shape = {1};
    auto max_idx_t = Tensor::create(one_shape, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
    auto max_val_t = Tensor::create(one_shape, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
    LlaisysTensor *max_idx = new LlaisysTensor{max_idx_t};
    LlaisysTensor *max_val = new LlaisysTensor{max_val_t};
    std::cerr << "[llaisys qwen2] calling argmax" << std::endl;
    llaisysArgmax(max_idx, max_val, logits);
    std::cerr << "[llaisys qwen2] after argmax" << std::endl;

    int64_t *res_ptr = reinterpret_cast<int64_t *>(max_idx->tensor->data());
    int64_t next = res_ptr ? res_ptr[0] : model->meta.end_token;

    // cleanup
    delete idx;
    delete x;
    delete logits;
    delete max_idx;
    delete max_val;

    return next;
}

} // extern "C"

// end of file
