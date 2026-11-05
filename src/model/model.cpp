#include "model/model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/error.h"
#include "core/placement.h"
#include "formats/dequant.h"
#include "kernels/attention/attention.h"
#include "kernels/deltanet/deltanet.h"
#include "kernels/gemv/gemv.h"
#include "kernels/hc/hc.h"
#include "kernels/moe/moe.h"
#include "kernels/norm/norm.h"
#include "kernels/ple/ple.h"
#include "kernels/sampling/argmax.h"
#include "model/weights.h"
#include "runtime/direct_io.h"
#include "runtime/expert_store.h"
#include "runtime/ple_rows.h"

namespace torch::model {
namespace {

using kernels::BlockQ8;

constexpr int kReadThreads = 16;  // 128 KiB reads reach ~5.4 GB/s at queue depth 16 (bench/ssd_reads.cpp)

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
    float *q, *k, *v, *attn, *attn_scratch;
    float *router, *weights, *h, *y, *shared_g, *shared_u, *shared_h, *shared_y, *shared_gate;
    int32_t* ids;
    float *key, *key_n, *value, *query, *ple_gate, *gated, *ple_x;
    float* logits;
    kernels::ArgmaxResult *argmax, *argmax_scratch;
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
        attn_scratch = d.take<float>(static_cast<size_t>(kernels::attention_scratch_floats(c.n_head, context)));
        router = d.take<float>(static_cast<size_t>(c.n_expert)), weights = d.take<float>(K);
        ids = d.take<int32_t>(K), h = d.take<float>(K * FF), y = d.take<float>(K * E);
        const size_t SH = static_cast<size_t>(c.n_ff_shared);
        shared_g = d.take<float>(SH), shared_u = d.take<float>(SH), shared_h = d.take<float>(SH);
        shared_y = d.take<float>(E), shared_gate = d.take<float>(1);
        key = d.take<float>(HCD), key_n = d.take<float>(HCD), value = d.take<float>(E), query = d.take<float>(HCD);
        ple_gate = d.take<float>(static_cast<size_t>(c.hc)), gated = d.take<float>(HCD), ple_x = d.take<float>(HCD);
        logits = d.take<float>(static_cast<size_t>(c.n_vocab));
        argmax = d.take<kernels::ArgmaxResult>(1),
        argmax_scratch = d.take<kernels::ArgmaxResult>(kernels::kArgmaxBlocks);
        q8 = d.take<BlockQ8>(HCD / 32), q8_h = d.take<BlockQ8>(K * FF / 32);
    }
};

// Pinned host memory the GPU copies to and from.
struct HostBuffers {
    float* logits;
    float* inputs;  // the token embedding, then the PLE rows
    kernels::ArgmaxResult* argmax;
    int32_t* pos;

    static uint64_t bytes(const Config& c) {
        return (2 * static_cast<uint64_t>(c.n_embd) + static_cast<uint64_t>(c.n_vocab)) * sizeof(float) +
               sizeof(kernels::ArgmaxResult) + sizeof(int32_t);
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
    RouteCounts ranking;  // the order experts are made resident in, without --resident
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
    // dir/<position>-<layer>-<stage>.f32. TORCH_DUMP_STAGES=a,b limits both to those stages. Slow: every dumped stage
    // waits for the GPU.
    const char* dump_dir = std::getenv("TORCH_DUMP_DIR");
    const std::string dump_stages =
        std::getenv("TORCH_DUMP_STAGES") ? std::string(",") + std::getenv("TORCH_DUMP_STAGES") + "," : "";
    const bool dumping = std::getenv("TORCH_DUMP") != nullptr || dump_dir != nullptr;
    // A token runs as one replayed CUDA graph; TORCH_EAGER=1 (and the stage dumps) launch kernel by kernel instead.
    const bool eager = dumping || std::getenv("TORCH_EAGER") != nullptr;
    cudaGraphExec_t graph = nullptr;
    bool want_logits = true;
    int layer_ = -1;  // the layer being computed, for dump lines

    explicit Impl(const Options& o)
        : shard1(gguf::File::open(o.shard1)),
          shard2(gguf::File::open(o.shard2)),
          file1(o.shard1),
          file2(o.shard2),
          cfg(Config::read(shard1)),
          context(o.context) {
        want_logits = o.logits;
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
            {"expert staging (" + std::to_string(cfg.n_expert_used) + " slots), map and tables",
             runtime::ExpertStore::fixed_bytes(cfg, shard1)},
            {"CUDA context and allocator slack", kCudaContext},
        };
        budget = plan_budget(mem_available(), kFreeFloor, std::move(reservations));
        const Placement placement = place(o);

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
        hb.argmax = reinterpret_cast<kernels::ArgmaxResult*>(hb.inputs + 2 * cfg.n_embd);
        hb.pos = reinterpret_cast<int32_t*>(hb.argmax + 1);

        weights = std::make_unique<Weights>(cfg, shard1, file1, pool);
        store = std::make_unique<runtime::ExpertStore>(cfg, weights->layers, file1, placement.resident);
        ple = std::make_unique<runtime::PleRows>(cfg, shard2, file2);
        if (!o.resident)
            top_up(placement);
        reset();
    }

    // Measures MemAvailable now that everything is allocated and makes the next-ranked experts resident until it is
    // down to the floor plus kTopUpMargin (budget.h). Output does not change, only the hit rate.
    void top_up(const Placement& placement) {
        budget.available_after_load = mem_available();
        const uint64_t keep = budget.floor + kTopUpMargin;
        if (budget.available_after_load <= keep)
            return;
        const Placement more = extend_ranked(ranking, runtime::ExpertStore::layer_expert_bytes(cfg, shard1), placement,
                                             budget.available_after_load - keep);
        store->add(more.resident);
        budget.added = more.count();
        budget.added_bytes = more.bytes;
    }

    // Which experts are resident: exactly --resident per layer, the most-routed in Options::placement, or as many of
    // every layer's first experts as fit. Fills in the budget's placement lines and the ranking top_up() continues.
    Placement place(const Options& o) {
        const std::vector<uint64_t> bytes = runtime::ExpertStore::layer_expert_bytes(cfg, shard1);
        Placement p;
        if (o.resident) {
            if (*o.resident < 0 || *o.resident > cfg.n_expert)
                throw Error("--resident must be between 0 and " + std::to_string(cfg.n_expert) + ".");
            p = place_uniform(*o.resident, cfg.n_expert, bytes);
            if (p.bytes > budget.room())
                throw Error("Only " + std::to_string(uniform_capacity(budget.room(), cfg.n_expert, bytes)) +
                            " experts per layer fit in free memory, not " + std::to_string(*o.resident) +
                            "; ask for fewer with --resident.");
            budget.placement = *o.resident == 0
                                   ? "no experts resident (--resident 0)"
                                   : "experts 0 to " + std::to_string(*o.resident - 1) + " of every layer (--resident)";
        } else if (!o.placement.empty()) {
            const RouteCounts counts = RouteCounts::read(o.placement);
            if (counts.n_layer != cfg.n_layer || counts.n_expert != cfg.n_expert)
                throw Error("The route counts in " + o.placement.string() + " were recorded with another model.");
            p = place_ranked(counts, bytes, budget.room());
            budget.placement = "the most-routed experts in " + o.placement.string();
            ranking = counts;
        } else {
            const int k = uniform_capacity(budget.room(), cfg.n_expert, bytes);
            p = place_uniform(k, cfg.n_expert, bytes);
            budget.placement = "experts 0 to " + std::to_string(k - 1) + " of every layer (no route counts given)";
            ranking = uniform_ranking(cfg.n_layer, cfg.n_expert);
        }
        budget.resident = p.count();
        budget.experts = cfg.n_layer * cfg.n_expert;
        budget.resident_bytes = p.bytes;
        return p;
    }

    ~Impl() {
        if (stream)
            cudaStreamSynchronize(stream);
        if (graph)
            cudaGraphExecDestroy(graph);
        if (device)
            cudaFree(device);
        if (host)
            cudaFreeHost(host);
        if (stream)
            cudaStreamDestroy(stream);
    }

    void dump(const char* stage, int layer, const float* d, int n) {
        if (!dumping || (!dump_stages.empty() && dump_stages.find(std::string(",") + stage + ",") == std::string::npos))
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
    void matvec(const Matrix& m, const float* x, float* y, bool quantized = false) {
        products(x, {{&m, y}}, quantized);
    }

    // y_i = m_i . x for matrices that read the same x (at most 4, one width): x is quantized once into b.q8 for the
    // quantized ones, which share one launch when gemv_q8_group takes their formats; two BF16 ones share one too.
    // `quantized`: the kernel that wrote x also wrote it to b.q8 (hc_collapse for b.mixed, rms_norm_q8), so the
    // products of a block's b.mixed run before any product of another vector replaces b.q8.
    struct Product {
        const Matrix* m;
        float* y;
    };
    void products(const float* x, std::initializer_list<Product> list, bool quantized = false) {
        Product bf16[4], q8[4];
        kernels::Q8Part parts[4];
        int nb = 0, nq = 0, np = 0;
        for (const Product& p : list)
            (p.m->type == Type::BF16 ? bf16[nb++] : q8[nq++]) = p;
        if (nq > 0 && !quantized)
            kernels::quantize_q8(x, q8[0].m->cols, b.q8, stream);
        for (int i = 0; i < nq; ++i) {
            const Matrix& m = *q8[i].m;
            if (nq > 1 && kernels::gemv_q8_groups(m.type))
                parts[np++] = {m.type, m.data, m.rows, q8[i].y};
            else
                kernels::gemv_q8(m.type, m.data, m.rows, m.cols, b.q8, q8[i].y, stream);
        }
        if (np == 1)
            kernels::gemv_q8(parts[0].type, parts[0].w, parts[0].rows, q8[0].m->cols, b.q8, parts[0].y, stream);
        else if (np > 1)
            kernels::gemv_q8_group(parts, np, q8[0].m->cols, b.q8, false, stream);
        for (int i = 0; i < nb; i += 2) {
            const Matrix& m = *bf16[i].m;
            if (i + 1 < nb)
                kernels::gemv_bf16_pair(m.data, m.rows, bf16[i + 1].m->data, bf16[i + 1].m->rows, m.cols, x, bf16[i].y,
                                        bf16[i + 1].y, stream);
            else
                kernels::gemv_bf16(m.data, m.rows, m.cols, x, bf16[i].y, stream);
        }
    }

    // A hyper-connection mixer: the hc streams of the residual, normalized, mixed into one block input (b.mixed),
    // and, when `inject` is given, the block output's per-stream weights (b.inject). With `combine`, the previous
    // block's output (b.out, weighted by b.inject) is first added to the residual, in the same launch as the norm.
    void hc_mix(const float* norm, const Matrix& down, const Matrix& up, const Matrix* inject, bool combine) {
        const int E = cfg.n_embd;
        if (combine)
            kernels::hc_combine_norm(b.res, b.out, b.inject, norm, b.xn, E, cfg.hc, cfg.eps, stream);
        else
            kernels::rms_norm(b.res, E, norm, E, nullptr, 0, b.xn, E, E, cfg.hc, cfg.eps, stream);
        // A BF16 down projection applies the low-rank SiLU as it stores; BF16 inject rows read the same input, so
        // they share its launch.
        const float scale = 1.0f / static_cast<float>(cfg.hc);
        const bool paired =
            inject && down.type == Type::BF16 && inject->type == Type::BF16 && inject->cols == down.cols;
        if (down.type == Type::BF16) {
            kernels::gemv_bf16_pair(down.data, down.rows, paired ? inject->data : nullptr, paired ? inject->rows : 0,
                                    down.cols, b.xn, b.lo, b.inject, stream, scale);
        } else {
            matvec(down, b.xn, b.lo);
            kernels::silu_scale(b.lo, cfg.hc_rank, scale, stream);
        }
        matvec(up, b.lo, b.hc_gate);
        kernels::hc_collapse(b.xn, b.hc_gate, b.mixed, b.q8, E, cfg.hc, stream);
        if (inject && !paired)
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
        products(b.mixed, {{&L.qkv, b.qkv}, {&L.z, b.z}, {&L.beta, b.beta}, {&L.alpha, b.alpha}}, true);
        dump("qkv", layer_, b.qkv, cfg.conv_channels());
        dump("z", layer_, b.z, HV * DS);
        dump("beta_alpha", layer_, b.beta, HV);
        // q and k are L2-normalized in the convolution's launch; the gates are computed in gdn_step's
        kernels::conv_silu_l2(b.conv_state[l], b.qkv, L.conv, b.conv_out, cfg.conv_channels(), 2 * QK, cfg.eps, stream);
        dump("conv_w", layer_, L.conv, cfg.ssm_conv * cfg.conv_channels());
        dump("conv_l2", layer_, b.conv_out, cfg.conv_channels());
        kernels::gdn_step(b.ssm_state[l], b.conv_out, b.conv_out + QK, b.conv_out + 2 * QK, b.alpha, b.beta, L.dt_bias,
                          L.a, b.o, cfg.ssm_k_heads, HV, 1.0f / std::sqrt(static_cast<float>(DS)), stream);
        dump("gdn_beta", layer_, b.beta, HV);
        dump("gdn_g", layer_, b.alpha, HV);
        dump("gdn_out", layer_, b.o, HV * DS);
        kernels::rms_norm_q8(b.o, DS, L.ssm_norm, 0, b.z, DS, b.o, DS, b.q8, DS, HV, cfg.eps, stream);
        dump("gdn_norm", layer_, b.o, HV * DS);
        matvec(L.ssm_out, b.o, b.out, true);
        dump("linear_attn_out", layer_, b.out, cfg.n_embd);
    }

    void attention(const Layer& L, int il) {
        const int D = cfg.head_dim, H = cfg.n_head, HKV = cfg.n_head_kv;
        const size_t l = static_cast<size_t>(il);
        products(b.mixed, {{&L.wq, b.q}, {&L.wk, b.k}, {&L.wv, b.v}}, true);  // q: per head [q | gate]
        kernels::qk_norm_rope_store(b.q, 2 * D, L.q_norm, b.k, L.k_norm, b.v, b.k_cache[l], b.v_cache[l], H, HKV,
                                    cfg.n_rot, b.pos, cfg.rope_base, cfg.eps, stream);
        kernels::attention_decode(b.q, b.q + D, 2 * D, b.k_cache[l], b.v_cache[l], b.attn, b.attn_scratch, b.pos, H,
                                  HKV, 1.0f / std::sqrt(static_cast<float>(D)), context, stream);
        dump("attn", layer_, b.attn, H * D);
        matvec(L.wo, b.attn, b.out);
        dump("attn_out", layer_, b.out, cfg.n_embd);
    }

    void experts(const Layer& L, int il) {
        const int E = cfg.n_embd, K = cfg.n_expert_used, FF = cfg.n_ff_expert;
        products(b.mixed, {{&L.router, b.router}, {&L.shared_gate_inp, b.shared_gate}}, true);
        dump("router", layer_, b.router, cfg.n_expert);
        kernels::router_top_k(b.router, cfg.n_expert, K, b.ids, b.weights, stream);
        // The layer's expert pointer table is written on the GPU; misses are read by the store's I/O thread.
        const int slots = store->slots();
        const uint8_t** table = store->table(il);
        runtime::expert_gate(b.ids, K, il, store->map(), cfg.n_expert, table, slots, store->mailbox(), store->seq(),
                             store->counters(), store->route_counts(), store->zeros(), stream);

        // the routed and the shared experts' gate and up read b.mixed as hc_collapse quantized it
        kernels::gemv_q8_experts(L.gate_exps.type, table, table + slots, K, FF, E, b.q8, 0, b.h, stream);
        const Matrix &sg = L.shared_gate, &su = L.shared_up;
        if (kernels::gemv_q8_groups(sg.type) && kernels::gemv_q8_groups(su.type) && sg.rows == su.rows) {
            const kernels::Q8Part parts[2] = {{sg.type, sg.data, sg.rows, b.shared_h},
                                              {su.type, su.data, su.rows, nullptr}};
            kernels::gemv_q8_group(parts, 2, E, b.q8, true, stream);
        } else {
            kernels::gemv_q8(sg.type, sg.data, sg.rows, E, b.q8, b.shared_g, stream);
            kernels::gemv_q8(su.type, su.data, su.rows, E, b.q8, b.shared_u, stream);
            kernels::swiglu(b.shared_g, b.shared_u, b.shared_h, cfg.n_ff_shared, stream);
        }
        // b.h and b.shared_h are quantized by their own launches: writing them as Q8 from gate and up measured slower
        // (+5-9 us a routed launch, +0.15-0.3 ms a token against 0.14 ms of quantize_q8; branch m3-gemv-q8-epilogues).
        kernels::quantize_q8(b.h, int64_t{K} * FF, b.q8_h, stream);
        kernels::gemv_q8_experts(L.down_exps.type, table + 2 * slots, nullptr, K, E, FF, b.q8_h, FF / 32, b.y, stream);
        matvec(L.shared_down, b.shared_h, b.shared_y);
        kernels::moe_combine(b.y, b.weights, K, b.shared_y, b.shared_gate, b.out, E, stream);
        dump("experts_h", layer_, b.h, K * FF);
        dump("experts_y", layer_, b.y, K * E);
        dump("shared_y", layer_, b.shared_y, E);
        dump("ffn_out", layer_, b.out, E);
    }

    // Everything the GPU does for one token, on `stream`: run directly, or recorded once into the token's graph.
    void enqueue() {
        const int E = cfg.n_embd;
        TORCH_CUDA_CHECK(cudaMemcpyAsync(b.pos, hb.pos, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        TORCH_CUDA_CHECK(
            cudaMemcpyAsync(b.embd, hb.inputs, sizeof(float) * static_cast<size_t>(E), cudaMemcpyHostToDevice, stream));
        TORCH_CUDA_CHECK(cudaMemcpyAsync(b.ple_embd, hb.inputs + E, sizeof(float) * static_cast<size_t>(E),
                                         cudaMemcpyHostToDevice, stream));

        kernels::hc_broadcast(b.embd, b.res, E, cfg.hc, stream);
        layer_ = -1;
        dump("embd", layer_, b.embd, E);
        dump("ple_rows", layer_, b.ple_embd, E);
        // A block's output is added to the residual in the next mixer's norm launch (hc_combine_norm), except before a
        // PLE block, which reads the residual first, and when dumping stages, which shows the residual per layer.
        bool pending = false;  // b.out and b.inject still to be added to b.res
        const auto settle = [&] {
            if (pending)
                kernels::hc_combine(b.res, b.out, b.inject, E, cfg.hc, stream);
            pending = false;
        };
        for (int il = 0; il < cfg.n_layer; ++il) {
            const Layer& L = weights->layers[static_cast<size_t>(il)];
            layer_ = il;
            if (L.ple) {
                settle();
                ple_block(L);
            }
            hc_mix(L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, &L.hc_attn_inject, pending);
            if (L.recurrent)
                deltanet(L, il);
            else
                attention(L, il);
            hc_mix(L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up, &L.hc_ffn_inject, true);
            experts(L, il);
            pending = true;
            if (dumping) {
                settle();
                dump("residual", layer_, b.res, cfg.hc * E);
            }
        }
        layer_ = -1;
        // The head's mixer is the output norm: there is no separate one.
        hc_mix(weights->head_hc_norm, weights->head_hc_down, weights->head_hc_up, nullptr, pending);
        matvec(weights->output, b.mixed, b.logits, true);
        dump("logits", layer_, b.logits, cfg.n_vocab);
        kernels::argmax(b.logits, cfg.n_vocab, b.argmax_scratch, b.argmax, stream);
        TORCH_CUDA_CHECK(
            cudaMemcpyAsync(hb.argmax, b.argmax, sizeof(kernels::ArgmaxResult), cudaMemcpyDeviceToHost, stream));
        if (want_logits)
            TORCH_CUDA_CHECK(cudaMemcpyAsync(hb.logits, b.logits, sizeof(float) * static_cast<size_t>(cfg.n_vocab),
                                             cudaMemcpyDeviceToHost, stream));
    }

    void decode(int32_t token) {
        const auto start = Clock::now();
        if (token < 0 || token >= cfg.n_vocab)
            throw Error("Token id " + std::to_string(token) + " is outside the vocabulary of " +
                        std::to_string(cfg.n_vocab) + ".");
        if (static_cast<int>(history.size()) >= context)
            throw Error("The context of " + std::to_string(context) + " tokens is full; start a new sequence.");
        const int E = cfg.n_embd;
        const size_t pos = history.size();
        history.push_back(token);

        // The token embedding and the PLE rows are single rows: the CPU dequantizes them to F32 into pinned memory,
        // which the token's first copies read.
        const Matrix& te = weights->token_embd;
        dequant_row(te.type, te.data + static_cast<uint64_t>(token) * te.row_bytes(), E, hb.inputs);
        const auto ple_start = Clock::now();
        ple->read(history, pos, hb.inputs + E);
        const double ple_ms = ms(Clock::now() - ple_start);
        times.ple += ple_ms;
        times.ple_max = std::max(times.ple_max, ple_ms);
        times.ple_slow += ple_ms > 5.0 ? 1 : 0;
        *hb.pos = static_cast<int32_t>(pos);

        if (eager) {
            enqueue();
        } else {
            if (graph == nullptr) {  // recorded at the first token; every pointer it bakes in stays fixed
                cudaGraph_t g = nullptr;
                TORCH_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                try {
                    enqueue();
                } catch (...) {
                    cudaStreamEndCapture(stream, &g);  // leave the stream usable; the half-recorded graph is dropped
                    if (g)
                        cudaGraphDestroy(g);
                    throw;
                }
                TORCH_CUDA_CHECK(cudaStreamEndCapture(stream, &g));
                const cudaError_t made = cudaGraphInstantiate(&graph, g, 0);
                cudaGraphDestroy(g);
                TORCH_CUDA_CHECK(made);
            }
            TORCH_CUDA_CHECK(cudaGraphLaunch(graph, stream));
        }
        const auto gpu_start = Clock::now();
        TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));
        TORCH_CUDA_CHECK(cudaGetLastError());
        store->check();
        times.gpu += ms(Clock::now() - gpu_start);
        times.total += ms(Clock::now() - start);
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
    DecodeTimes t = impl_->times;
    t.expert_reads = impl_->store->read_ms();
    t.ple_hedged = impl_->ple->hedged_reads();
    return t;
}

ExpertStats Model::expert_stats() const {
    const auto [hits, misses] = impl_->store->hits_and_misses();
    return {impl_->budget.resident + impl_->budget.added, impl_->budget.experts, hits, misses,
            impl_->store->hedged_reads()};
}

RouteCounts Model::route_counts() const {
    RouteCounts c;
    c.n_layer = impl_->cfg.n_layer;
    c.n_expert = impl_->cfg.n_expert;
    c.counts = impl_->store->read_route_counts();
    return c;
}

int Model::position() const noexcept {
    return static_cast<int>(impl_->history.size());
}

bool Model::decode(int32_t token) noexcept {
    try {
        impl_->error.clear();
        impl_->decode(token);
        return true;
    } catch (const std::exception& e) {
        impl_->error = e.what();
        return false;
    }
}

std::span<const float> Model::logits() const noexcept {
    if (!impl_->want_logits)
        return {};
    return {impl_->hb.logits, static_cast<size_t>(impl_->cfg.n_vocab)};
}

int32_t Model::argmax() const noexcept {
    return impl_->hb.argmax->index;
}

const std::string& Model::error() const noexcept {
    return impl_->error;
}

void Model::reset() {
    impl_->reset();
}

}  // namespace torch::model
