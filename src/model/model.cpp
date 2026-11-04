#include "model/model.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/error.h"
#include "formats/dequant.h"
#include "kernels/attention/attention.h"
#include "kernels/deltanet/deltanet.h"
#include "kernels/gemv/gemv.h"
#include "kernels/hc/hc.h"
#include "kernels/moe/moe.h"
#include "kernels/norm/norm.h"
#include "kernels/ple/ple.h"
#include "model/weights.h"
#include "runtime/direct_io.h"
#include "runtime/expert_store.h"
#include "runtime/ple_rows.h"

namespace torch::model {
namespace {

using kernels::BlockQ8;

constexpr int kReadThreads = 8;

using Clock = std::chrono::steady_clock;
double ms(Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}
constexpr uint64_t kCudaContext = kGiB / 2;  // the CUDA context and allocator slack [est]

// Lays out device buffers in one allocation. The dry run (no base) only adds up the bytes, so the memory budget and
// the allocation come from the same list.
class DeviceLayout {
public:
    explicit DeviceLayout(uint8_t* base) : base_(base) {}

    template <class T>
    T* take(size_t count) {
        T* at = base_ ? reinterpret_cast<T*>(base_ + used_) : nullptr;
        used_ += (count * sizeof(T) + 255) / 256 * 256;
        return at;
    }
    uint64_t used() const noexcept { return used_; }

private:
    uint8_t* base_;
    uint64_t used_ = 0;
};

// Everything the forward pass keeps on the GPU. The sequence state comes first, so reset() clears one prefix.
struct Buffers {
    // Sequence state.
    std::vector<uint16_t*> k_cache, v_cache;    // attention layers: [context][kv heads x head dim], F16
    std::vector<float*> conv_state, ssm_state;  // DeltaNet layers: [3][channels]; [v heads][128][128]
    float* ple_history = nullptr;               // [9][hc x n_embd]
    uint64_t state_bytes = 0;

    int32_t* pos = nullptr;  // the position being computed, read by the kernels
    float *res, *xn, *lo, *hc_gate, *mixed, *inject, *out;
    float *embd, *ple_embd;
    float *qkv, *z, *beta, *alpha, *conv_out, *o;
    float *q, *k, *v, *attn;
    float *router, *weights, *h, *y, *shared_g, *shared_u, *shared_h, *shared_y, *shared_gate;
    int32_t* ids;
    float *key, *key_n, *value, *query, *ple_gate, *gated, *ple_x;
    float* logits;
    BlockQ8 *q8, *q8_h;

    void lay_out(DeviceLayout& d, const Config& c, int context) {
        const size_t E = static_cast<size_t>(c.n_embd), HCD = static_cast<size_t>(c.hc) * E;
        const size_t KV = static_cast<size_t>(c.n_head_kv * c.head_dim);
        const size_t CC = static_cast<size_t>(c.conv_channels());
        const size_t HV = static_cast<size_t>(c.ssm_v_heads), DS = static_cast<size_t>(c.ssm_head_dim);
        const size_t V = HV * DS, K = static_cast<size_t>(c.n_expert_used), FF = static_cast<size_t>(c.n_ff_expert);
        k_cache.assign(static_cast<size_t>(c.n_layer), nullptr);
        v_cache = k_cache;
        conv_state.assign(static_cast<size_t>(c.n_layer), nullptr);
        ssm_state = conv_state;
        for (int il = 0; il < c.n_layer; ++il) {
            const size_t l = static_cast<size_t>(il);
            if (c.recurrent(il)) {
                conv_state[l] = d.take<float>(3 * CC);
                ssm_state[l] = d.take<float>(HV * DS * DS);
            } else {
                k_cache[l] = d.take<uint16_t>(static_cast<size_t>(context) * KV);
                v_cache[l] = d.take<uint16_t>(static_cast<size_t>(context) * KV);
            }
        }
        ple_history = d.take<float>(9 * HCD);
        state_bytes = d.used();

        pos = d.take<int32_t>(1);
        res = d.take<float>(HCD), xn = d.take<float>(HCD), lo = d.take<float>(static_cast<size_t>(c.hc_rank));
        hc_gate = d.take<float>(HCD), mixed = d.take<float>(E), inject = d.take<float>(static_cast<size_t>(c.hc));
        out = d.take<float>(E), embd = d.take<float>(E), ple_embd = d.take<float>(E);
        qkv = d.take<float>(CC), z = d.take<float>(V), beta = d.take<float>(HV), alpha = d.take<float>(HV);
        conv_out = d.take<float>(CC), o = d.take<float>(V);
        q = d.take<float>(2 * static_cast<size_t>(c.n_head * c.head_dim)), k = d.take<float>(KV);
        v = d.take<float>(KV), attn = d.take<float>(static_cast<size_t>(c.n_head * c.head_dim));
        router = d.take<float>(static_cast<size_t>(c.n_expert)), weights = d.take<float>(K);
        ids = d.take<int32_t>(K), h = d.take<float>(K * FF), y = d.take<float>(K * E);
        const size_t SH = static_cast<size_t>(c.n_ff_shared);
        shared_g = d.take<float>(SH), shared_u = d.take<float>(SH), shared_h = d.take<float>(SH);
        shared_y = d.take<float>(E), shared_gate = d.take<float>(1);
        key = d.take<float>(HCD), key_n = d.take<float>(HCD), value = d.take<float>(E), query = d.take<float>(HCD);
        ple_gate = d.take<float>(static_cast<size_t>(c.hc)), gated = d.take<float>(HCD), ple_x = d.take<float>(HCD);
        logits = d.take<float>(static_cast<size_t>(c.n_vocab));
        q8 = d.take<BlockQ8>(HCD / 32), q8_h = d.take<BlockQ8>(K * FF / 32);
    }
};

// Pinned host memory the GPU copies to and from.
struct HostBuffers {
    float* inputs;  // the token embedding, then the PLE rows
    int32_t* ids;   // a layer's routing
    int32_t* pos;
    float* logits;

    static uint64_t bytes(const Config& c) {
        return (2 * static_cast<uint64_t>(c.n_embd) + static_cast<uint64_t>(c.n_vocab)) * sizeof(float) +
               (static_cast<uint64_t>(c.n_expert_used) + 1) * sizeof(int32_t);
    }
};

}  // namespace

struct Model::Impl {
    gguf::File shard1, shard2;
    runtime::DirectFile file1, file2;
    Config cfg;
    int context;
    runtime::ReadPool pool{kReadThreads};
    Budget budget;
    std::unique_ptr<Weights> weights;
    std::unique_ptr<runtime::ExpertStore> store;
    std::unique_ptr<runtime::PleRows> ple;
    cudaStream_t stream = nullptr;
    uint8_t* device = nullptr;
    uint8_t* host = nullptr;
    Buffers b;
    HostBuffers hb{};
    std::vector<int32_t> history;
    std::string error;
    DecodeTimes times;
    // TORCH_DUMP=1: after every stage, one line on stderr with the stage's sum, mean |x| and count of non-finite
    // values, for bisecting numerics. TORCH_DUMP_DIR=dir also writes each stage's values to
    // dir/<position>-<layer>-<stage>.f32. Slow: every stage waits for the GPU.
    const char* dump_dir = std::getenv("TORCH_DUMP_DIR");
    const bool dumping = std::getenv("TORCH_DUMP") != nullptr || dump_dir != nullptr;
    int layer_ = -1;  // the layer being computed, for dump lines

    explicit Impl(const Options& o)
        : shard1(gguf::File::open(o.shard1)),
          shard2(gguf::File::open(o.shard2)),
          file1(o.shard1),
          file2(o.shard2),
          cfg(Config::read(shard1)),
          context(o.context) {
        if (context < 1 || context > cfg.indexer_top_k)
            throw Error("The context must be between 1 and " + std::to_string(cfg.indexer_top_k) +
                        " tokens in this version, where attention is computed densely.");

        // MemAvailable is read before the CUDA context exists; the budget reserves room for it.
        DeviceLayout dry(nullptr);
        b.lay_out(dry, cfg, context);
        std::vector<Reservation> reservations = {
            {"weights other than routed experts", Weights::base_bytes(cfg, shard1)},
            {"KV cache, DeltaNet state and activations (" + std::to_string(context) + " tokens)",
             dry.used() + HostBuffers::bytes(cfg)},
            {"expert staging (" + std::to_string(cfg.n_expert_used) + " slots) and tables",
             runtime::ExpertStore::staging_bytes(cfg, shard1, cfg.n_expert_used)},
            {"CUDA context and allocator slack", kCudaContext},
        };
        budget = plan_budget(mem_available(), kFreeFloor, std::move(reservations), Weights::expert_bytes(cfg, shard1),
                             cfg.n_expert, o.resident);

        TORCH_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        kernels::upload_iq_tables();
        void* d = nullptr;
        TORCH_CUDA_CHECK(cudaMalloc(&d, dry.used()));
        device = static_cast<uint8_t*>(d);
        DeviceLayout layout(device);
        b.lay_out(layout, cfg, context);
        void* h = nullptr;
        TORCH_CUDA_CHECK(cudaHostAlloc(&h, HostBuffers::bytes(cfg), cudaHostAllocDefault));
        host = static_cast<uint8_t*>(h);
        hb.logits = reinterpret_cast<float*>(host);
        hb.inputs = hb.logits + cfg.n_vocab;
        hb.ids = reinterpret_cast<int32_t*>(hb.inputs + 2 * cfg.n_embd);
        hb.pos = hb.ids + cfg.n_expert_used;

        weights = std::make_unique<Weights>(cfg, shard1, file1, budget.resident, pool);
        store = std::make_unique<runtime::ExpertStore>(cfg, weights->layers, file1, budget.resident, cfg.n_expert_used,
                                                       pool);
        ple = std::make_unique<runtime::PleRows>(cfg, shard2, file2, pool);
        reset();
    }

    ~Impl() {
        if (stream)
            cudaStreamSynchronize(stream);
        if (device)
            cudaFree(device);
        if (host)
            cudaFreeHost(host);
        if (stream)
            cudaStreamDestroy(stream);
    }

    void dump(const char* stage, int layer, const float* d, int n) {
        if (!dumping)
            return;
        std::vector<float> h(static_cast<size_t>(n));
        TORCH_CUDA_CHECK(cudaMemcpyAsync(h.data(), d, sizeof(float) * h.size(), cudaMemcpyDeviceToHost, stream));
        TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (dump_dir) {
            const std::string path = std::string(dump_dir) + "/" + std::to_string(history.size() - 1) + "-" +
                                     std::to_string(layer) + "-" + stage + ".f32";
            if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
                std::fwrite(h.data(), sizeof(float), h.size(), f);
                std::fclose(f);
            }
        }
        double sum = 0, abs = 0;
        int bad = 0;
        for (float v : h) {
            if (!std::isfinite(v)) {
                ++bad;
                continue;
            }
            sum += v;
            abs += std::fabs(v);
        }
        std::fprintf(stderr, "dump pos %zu layer %2d %-16s n %6d sum %14.6g mean|x| %12.6g non-finite %d [%g %g %g]\n",
                     history.size() - 1, layer, stage, n, sum, abs / n, bad, h[0], n > 1 ? h[1] : 0.0f,
                     n > 2 ? h[2] : 0.0f);
    }

    void reset() {
        history.clear();
        TORCH_CUDA_CHECK(cudaMemsetAsync(device, 0, b.state_bytes, stream));
        TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // y = m . x: BF16 weights take x as F32, quantized weights take it quantized to Q8 (gemv.h).
    void matvec(const Matrix& m, const float* x, float* y) {
        if (m.type == Type::BF16) {
            kernels::gemv_bf16(m.data, m.rows, m.cols, x, y, stream);
        } else {
            kernels::quantize_q8(x, m.cols, b.q8, stream);
            kernels::gemv_q8(m.type, m.data, m.rows, m.cols, b.q8, y, stream);
        }
    }

    // A hyper-connection mixer: the hc streams of the residual, normalized, mixed into one block input (b.mixed),
    // and, when `inject` is given, the block output's per-stream weights (b.inject).
    void hc_mix(const float* norm, const Matrix& down, const Matrix& up, const Matrix* inject) {
        const int E = cfg.n_embd;
        kernels::rms_norm(b.res, E, norm, E, nullptr, 0, b.xn, E, E, cfg.hc, cfg.eps, stream);
        matvec(down, b.xn, b.lo);
        kernels::silu_scale(b.lo, cfg.hc_rank, 1.0f / static_cast<float>(cfg.hc), stream);
        matvec(up, b.lo, b.hc_gate);
        kernels::hc_collapse(b.xn, b.hc_gate, b.mixed, E, cfg.hc, stream);
        if (inject)
            matvec(*inject, b.xn, b.inject);
        dump("hc_norm", layer_, b.xn, cfg.hc * E);
        dump("hc_lo", layer_, b.lo, cfg.hc_rank);
        dump("hc_mixed", layer_, b.mixed, E);
        if (inject)
            dump("hc_inject", layer_, b.inject, cfg.hc);
    }

    void ple_block(const Layer& L) {
        const int E = cfg.n_embd;
        matvec(L.ple_key, b.ple_embd, b.key);
        matvec(L.ple_value, b.ple_embd, b.value);
        kernels::rms_norm(b.key, E, L.ple_norm_key, E, nullptr, 0, b.key_n, E, E, cfg.hc, cfg.eps, stream);
        kernels::rms_norm(b.res, E, L.ple_norm_query, E, nullptr, 0, b.query, E, E, cfg.hc, cfg.eps, stream);
        kernels::ple_gate(b.key_n, b.query, b.ple_gate, E, cfg.hc, stream);
        kernels::ple_gated_value(b.value, b.ple_gate, b.gated, E, cfg.hc, stream);
        kernels::rms_norm(b.gated, E, L.ple_norm_conv, E, nullptr, 0, b.ple_x, E, E, cfg.hc, cfg.eps, stream);
        kernels::ple_conv(b.res, b.gated, b.ple_x, b.ple_history, L.ple_conv, cfg.hc * E, b.pos, stream);
        dump("ple_gate", layer_, b.ple_gate, cfg.hc);
        dump("ple_out", layer_, b.res, cfg.hc * E);
    }

    void deltanet(const Layer& L, int il) {
        const int DS = cfg.ssm_head_dim, QK = cfg.ssm_k_heads * DS, HV = cfg.ssm_v_heads;
        const size_t l = static_cast<size_t>(il);
        matvec(L.qkv, b.mixed, b.qkv);
        matvec(L.z, b.mixed, b.z);
        matvec(L.beta, b.mixed, b.beta);
        matvec(L.alpha, b.mixed, b.alpha);
        dump("qkv", layer_, b.qkv, cfg.conv_channels());
        dump("z", layer_, b.z, HV * DS);
        dump("beta_alpha", layer_, b.beta, HV);
        kernels::conv_silu(b.conv_state[l], b.qkv, L.conv, b.conv_out, cfg.conv_channels(), stream);
        dump("conv_w", layer_, L.conv, cfg.ssm_conv * cfg.conv_channels());
        dump("conv_raw", layer_, b.conv_out, cfg.conv_channels());
        kernels::l2_norm(b.conv_out, DS, b.conv_out, DS, DS, 2 * cfg.ssm_k_heads, cfg.eps, stream);  // q and k
        kernels::gdn_gates(b.beta, b.alpha, L.dt_bias, L.a, HV, stream);
        dump("gdn_beta", layer_, b.beta, HV);
        kernels::gdn_step(b.ssm_state[l], b.conv_out, b.conv_out + QK, b.conv_out + 2 * QK, b.alpha, b.beta, b.o,
                          cfg.ssm_k_heads, HV, 1.0f / std::sqrt(static_cast<float>(DS)), stream);
        dump("conv_l2", layer_, b.conv_out, cfg.conv_channels());
        dump("gdn_g", layer_, b.alpha, HV);
        dump("gdn_out", layer_, b.o, HV * DS);
        kernels::rms_norm(b.o, DS, L.ssm_norm, 0, b.z, DS, b.o, DS, DS, HV, cfg.eps, stream);
        dump("gdn_norm", layer_, b.o, HV * DS);
        matvec(L.ssm_out, b.o, b.out);
        dump("linear_attn_out", layer_, b.out, cfg.n_embd);
    }

    void attention(const Layer& L, int il) {
        const int D = cfg.head_dim, H = cfg.n_head, HKV = cfg.n_head_kv;
        const size_t l = static_cast<size_t>(il);
        matvec(L.wq, b.mixed, b.q);  // per head [q | gate]
        matvec(L.wk, b.mixed, b.k);
        matvec(L.wv, b.mixed, b.v);
        kernels::rms_norm(b.q, 2 * D, L.q_norm, 0, nullptr, 0, b.q, 2 * D, D, H, cfg.eps, stream);
        kernels::rms_norm(b.k, D, L.k_norm, 0, nullptr, 0, b.k, D, D, HKV, cfg.eps, stream);
        kernels::rope_neox(b.q, 2 * D, H, cfg.n_rot, b.pos, cfg.rope_base, stream);
        kernels::rope_neox(b.k, D, HKV, cfg.n_rot, b.pos, cfg.rope_base, stream);
        kernels::store_kv(b.k, b.v, b.k_cache[l], b.v_cache[l], HKV * D, b.pos, stream);
        kernels::attention_decode(b.q, b.q + D, 2 * D, b.k_cache[l], b.v_cache[l], b.attn, b.pos, H, HKV,
                                  1.0f / std::sqrt(static_cast<float>(D)), context, stream);
        dump("attn", layer_, b.attn, H * D);
        matvec(L.wo, b.attn, b.out);
        dump("attn_out", layer_, b.out, cfg.n_embd);
    }

    void experts(const Layer& L, int il) {
        const int E = cfg.n_embd, K = cfg.n_expert_used, FF = cfg.n_ff_expert;
        matvec(L.router, b.mixed, b.router);
        dump("router", layer_, b.router, cfg.n_expert);
        kernels::router_top_k(b.router, cfg.n_expert, K, b.ids, b.weights, stream);
        // The host needs the routing to place the experts: the one wait per layer (see model.h).
        TORCH_CUDA_CHECK(
            cudaMemcpyAsync(hb.ids, b.ids, sizeof(int32_t) * static_cast<size_t>(K), cudaMemcpyDeviceToHost, stream));
        const auto wait_start = Clock::now();
        TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto read_start = Clock::now();
        const runtime::ExpertStore::Tables t = store->prepare(il, {hb.ids, static_cast<size_t>(K)});
        times.routing += ms(read_start - wait_start);
        times.expert_reads += ms(Clock::now() - read_start);

        kernels::quantize_q8(b.mixed, E, b.q8, stream);
        kernels::gemv_q8_experts(L.gate_exps.type, t.gate, t.up, K, FF, E, b.q8, 0, b.h, stream);
        kernels::quantize_q8(b.h, int64_t{K} * FF, b.q8_h, stream);
        kernels::gemv_q8_experts(L.down_exps.type, t.down, nullptr, K, E, FF, b.q8_h, FF / 32, b.y, stream);

        matvec(L.shared_gate, b.mixed, b.shared_g);
        matvec(L.shared_up, b.mixed, b.shared_u);
        kernels::swiglu(b.shared_g, b.shared_u, b.shared_h, cfg.n_ff_shared, stream);
        matvec(L.shared_down, b.shared_h, b.shared_y);
        matvec(L.shared_gate_inp, b.mixed, b.shared_gate);
        kernels::moe_combine(b.y, b.weights, K, b.shared_y, b.shared_gate, b.out, E, stream);
        dump("experts_h", layer_, b.h, K * FF);
        dump("experts_y", layer_, b.y, K * E);
        dump("shared_y", layer_, b.shared_y, E);
        dump("ffn_out", layer_, b.out, E);
    }

    std::span<const float> decode(int32_t token) {
        const auto start = Clock::now();
        if (token < 0 || token >= cfg.n_vocab)
            throw Error("Token id " + std::to_string(token) + " is outside the vocabulary of " +
                        std::to_string(cfg.n_vocab) + ".");
        if (static_cast<int>(history.size()) >= context)
            throw Error("The context of " + std::to_string(context) + " tokens is full; start a new sequence.");
        const int E = cfg.n_embd;
        const size_t pos = history.size();
        history.push_back(token);

        // The token embedding and the PLE rows are single rows: the CPU dequantizes them to F32 and uploads them.
        const Matrix& te = weights->token_embd;
        dequant_row(te.type, te.data + static_cast<uint64_t>(token) * te.row_bytes(), E, hb.inputs);
        const auto ple_start = Clock::now();
        ple->read(history, pos, hb.inputs + E);
        times.ple += ms(Clock::now() - ple_start);
        *hb.pos = static_cast<int32_t>(pos);
        TORCH_CUDA_CHECK(cudaMemcpyAsync(b.pos, hb.pos, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        TORCH_CUDA_CHECK(
            cudaMemcpyAsync(b.embd, hb.inputs, sizeof(float) * static_cast<size_t>(E), cudaMemcpyHostToDevice, stream));
        TORCH_CUDA_CHECK(cudaMemcpyAsync(b.ple_embd, hb.inputs + E, sizeof(float) * static_cast<size_t>(E),
                                         cudaMemcpyHostToDevice, stream));

        kernels::hc_broadcast(b.embd, b.res, E, cfg.hc, stream);
        layer_ = -1;
        dump("embd", layer_, b.embd, E);
        dump("ple_rows", layer_, b.ple_embd, E);
        for (int il = 0; il < cfg.n_layer; ++il) {
            const Layer& L = weights->layers[static_cast<size_t>(il)];
            layer_ = il;
            if (L.ple)
                ple_block(L);
            hc_mix(L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, &L.hc_attn_inject);
            if (L.recurrent)
                deltanet(L, il);
            else
                attention(L, il);
            kernels::hc_combine(b.res, b.out, b.inject, E, cfg.hc, stream);
            hc_mix(L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up, &L.hc_ffn_inject);
            experts(L, il);
            kernels::hc_combine(b.res, b.out, b.inject, E, cfg.hc, stream);
            dump("residual", layer_, b.res, cfg.hc * E);
        }
        layer_ = -1;
        // The head's mixer is the output norm: there is no separate one.
        hc_mix(weights->head_hc_norm, weights->head_hc_down, weights->head_hc_up, nullptr);
        matvec(weights->output, b.mixed, b.logits);
        dump("logits", layer_, b.logits, cfg.n_vocab);
        TORCH_CUDA_CHECK(cudaMemcpyAsync(hb.logits, b.logits, sizeof(float) * static_cast<size_t>(cfg.n_vocab),
                                         cudaMemcpyDeviceToHost, stream));
        const auto finish_start = Clock::now();
        TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));
        TORCH_CUDA_CHECK(cudaGetLastError());
        times.finish += ms(Clock::now() - finish_start);
        times.total += ms(Clock::now() - start);
        return {hb.logits, static_cast<size_t>(cfg.n_vocab)};
    }
};

Model::Model(const Options& options) : impl_(std::make_unique<Impl>(options)) {}
Model::~Model() = default;

const Config& Model::config() const noexcept {
    return impl_->cfg;
}

const Budget& Model::budget() const noexcept {
    return impl_->budget;
}

DecodeTimes Model::decode_times() const noexcept {
    return impl_->times;
}

ExpertStats Model::expert_stats() const noexcept {
    return {impl_->store->resident(), impl_->store->hits(), impl_->store->misses()};
}

int Model::position() const noexcept {
    return static_cast<int>(impl_->history.size());
}

std::span<const float> Model::decode(int32_t token) noexcept {
    try {
        impl_->error.clear();
        return impl_->decode(token);
    } catch (const std::exception& e) {
        impl_->error = e.what();
        return {};
    }
}

const std::string& Model::error() const noexcept {
    return impl_->error;
}

void Model::reset() {
    impl_->reset();
}

}  // namespace torch::model
