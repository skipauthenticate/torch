// Kernel microbenchmark: how fast each GPU kernel of one decoded token runs, in GB/s against the streaming ceiling
// measured by bench/memory_bandwidth.cu. It is the baseline every kernel change is compared with (the kernel gate in
// tools/check_orin.sh).
//
// What it times. The tensor table of shard 1 gives the exact matrix-vector products a token runs (model.cpp): every
// non-expert matrix (gemv_q8 for quantized formats with its quantize_q8, gemv_bf16 for BF16, a mixer's inject rows in
// its down projection's launch, its up projection with hc_collapse in one gemv_bf16_collapse launch, and the
// matrices that read one input in the launches model.cpp groups them into:
// gemv_q8_group with one quantize_q8, BF16 pairs) and the routed experts (gemv_q8_experts with the model's slot count,
// gate and up fused, and down), each deduplicated by (kernel, formats, shapes) with the number of launches per token.
// With --columns T > 1 the matrices run on T columns at once and the routed gate+up and down on a window's unique
// experts (gemv_q8_experts_window and _down_window, each column keeping half its predecessor's experts).
// The small kernels (norms, hyper-connection helpers, DeltaNet, rotary embedding, attention, router, PLE) are timed
// once per token shape with their per-token counts.
//
// How. Weights are valid random blocks (finite F16 scales) in mapped pinned memory, as the engine allocates them.
// A matrix is allocated as enough distinct copies that one graph rotates through more than 64 MiB of weights, so the
// 4 MB L2 never holds them; experts are drawn 10 distinct at a time from a pool of slices through pointer tables in
// mapped memory, as ExpertStore does. Each case is captured as a CUDA graph of K launches (K = the rotation count, at
// least 20), replayed 20 times after one warm-up replay, timed with CUDA events: per launch = total / (K x 20).
// The small kernels run on one set of KB-sized buffers, so their inputs are warm in L2, as activations are in the
// engine; gdn_step and attention_decode rotate through copies of the state and the KV cache instead.
//
// Bytes. Matrices count their weight bytes, the Q8 or F32 activations they read and the F32 outputs they write; small
// kernels count every operand once (a stateful kernel reads and writes its state). attention_decode counts the K and
// V rows once, not once per query head. The timed kernels exclude what happens on the host and between launches:
// the token and PLE row reads, the routing wait, expert reads on a miss, the copies, and the gaps between launches
// in the real forward pass.
//
// usage: torch_bench_kernels MODEL-00001-of-0000N.gguf [--json FILE] [--ceiling GB/s, default 180] [--columns 1-8]
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "core/cuda_check.h"
#include "core/error.h"
#include "formats/types.h"
#include "gguf/gguf.h"
#include "kernels/attention/attention.h"
#include "kernels/deltanet/deltanet.h"
#include "kernels/gemv/gemv.h"
#include "kernels/hc/hc.h"
#include "kernels/moe/moe.h"
#include "kernels/norm/norm.h"
#include "kernels/ple/ple.h"
#include "model/config.h"
#include "model/weights.h"
#include "torch/version.h"

namespace torch::bench {
namespace {

using kernels::BlockQ8;

constexpr uint64_t kRotateBytes = 64ull << 20;  // one graph touches more weight bytes than this
constexpr int kMinLaunches = 20;
constexpr int kReplays = 20;
constexpr uint64_t kSmallBytes = 1ull << 20;  // the "small launch" threshold of the totals
constexpr int kAttentionContext = 2048;       // KV cache rows allocated
constexpr int kAttentionPosition = 1800;      // the position attention_decode is timed at

// ---- Small helpers ------------------------------------------------------------------------------------------------

class Rng {
public:
    uint64_t next() {
        state_ ^= state_ << 13, state_ ^= state_ >> 7, state_ ^= state_ << 17;  // xorshift64
        return state_;
    }
    float uniform(float lo, float hi) { return lo + (hi - lo) * static_cast<float>(next() >> 40) / 16777216.0f; }

private:
    uint64_t state_ = 0x9E3779B97F4A7C15ull;
};

template <class T>
class Device {
public:
    explicit Device(size_t count) : count_(count) { TORCH_CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T))); }
    Device(Device&& other) noexcept : data_(std::exchange(other.data_, nullptr)), count_(other.count_) {}
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device() { cudaFree(data_); }
    T* get() const { return data_; }
    size_t count() const { return count_; }
    void upload(const std::vector<T>& host, size_t at = 0) {
        TORCH_CUDA_CHECK(cudaMemcpy(data_ + at, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }

private:
    T* data_ = nullptr;
    size_t count_;
};

// Mapped pinned memory: written by the CPU, read by the GPU through device().
class Pinned {
public:
    explicit Pinned(size_t bytes) {
        void* p = nullptr;
        TORCH_CUDA_CHECK(cudaHostAlloc(&p, bytes, cudaHostAllocMapped));
        host_ = static_cast<uint8_t*>(p);
        void* d = nullptr;
        TORCH_CUDA_CHECK(cudaHostGetDevicePointer(&d, p, 0));
        device_ = static_cast<const uint8_t*>(d);
    }
    Pinned(const Pinned&) = delete;
    Pinned& operator=(const Pinned&) = delete;
    ~Pinned() { cudaFreeHost(host_); }
    uint8_t* host() const { return host_; }
    const uint8_t* device() const { return device_; }

private:
    uint8_t* host_ = nullptr;
    const uint8_t* device_ = nullptr;
};

Device<float> random_floats(size_t n, float lo, float hi, Rng& rng) {
    std::vector<float> v(n);
    for (float& x : v)
        x = rng.uniform(lo, hi);
    Device<float> d(n);
    d.upload(v);
    return d;
}

uint64_t round_up(uint64_t n, uint64_t to) {
    return (n + to - 1) / to * to;
}

// Copies needed so that one rotation passes more than kRotateBytes of `bytes`-sized pieces; at least two.
uint64_t copies_for(uint64_t bytes) {
    return std::max<uint64_t>(2, kRotateBytes / std::max<uint64_t>(bytes, 1) + 1);
}

// An F16 with a finite, moderate magnitude (0.0078 to 0.0156): the scale field of a random block.
uint16_t random_scale(Rng& rng) {
    return static_cast<uint16_t>(0x2000 | (rng.next() & 0x03FF));
}

// Where each format keeps its F16 scales (block offsets), so random blocks get finite ones.
std::vector<size_t> scale_offsets(Type t) {
    switch (t) {
        case Type::Q4_K:
        case Type::Q5_K:
            return {0, 2};
        case Type::Q6_K:
            return {208};
        default:
            return {0};
    }
}

// Fills `bytes` (a whole number of rows) with random blocks of `type` whose scales are finite.
void fill_weights(Type type, uint8_t* p, size_t bytes, Rng& rng) {
    for (size_t i = 0; i < bytes; i += 8) {
        const uint64_t v = rng.next();
        std::memcpy(p + i, &v, std::min<size_t>(8, bytes - i));
    }
    if (type == Type::BF16) {
        for (size_t i = 0; i + 2 <= bytes; i += 2) {
            uint16_t h;
            std::memcpy(&h, p + i, 2);
            h = static_cast<uint16_t>((h & 0x807F) | 0x3C00);  // keep sign and mantissa, exponent 2^-7
            std::memcpy(p + i, &h, 2);
        }
        return;
    }
    const size_t block = type_info(static_cast<uint32_t>(type))->block_bytes;
    for (size_t b = 0; b + block <= bytes; b += block) {
        for (size_t offset : scale_offsets(type)) {
            const uint16_t h = random_scale(rng);
            std::memcpy(p + b + offset, &h, 2);
        }
    }
}

Device<BlockQ8> random_q8(size_t blocks, Rng& rng) {
    std::vector<BlockQ8> v(blocks);
    for (BlockQ8& b : v) {
        int sum = 0;
        for (int8_t& q : b.q)
            q = static_cast<int8_t>(rng.next()), sum += q;
        b.d = random_scale(rng);
        b.sum = static_cast<int16_t>(sum);
    }
    Device<BlockQ8> d(blocks);
    d.upload(v);
    return d;
}

// Captures launch(0) .. launch(launches - 1) as one CUDA graph and returns microseconds per launch: the mean over
// kReplays replays that follow one warm-up replay. launch(0) also runs once eagerly first, so lazy loading of the
// kernels and attribute opt-ins stay out of the capture.
template <class Launch>
double time_graph(cudaStream_t stream, int launches, Launch&& launch) {
    launch(0);
    TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));
    TORCH_CUDA_CHECK(cudaGetLastError());

    cudaGraph_t graph = nullptr;
    TORCH_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < launches; ++i)
        launch(i);
    TORCH_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    TORCH_CUDA_CHECK(cudaGetLastError());
    cudaGraphExec_t exec = nullptr;
    TORCH_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));

    cudaEvent_t start, stop;
    TORCH_CUDA_CHECK(cudaEventCreate(&start));
    TORCH_CUDA_CHECK(cudaEventCreate(&stop));
    TORCH_CUDA_CHECK(cudaGraphLaunch(exec, stream));  // warm-up
    TORCH_CUDA_CHECK(cudaEventRecord(start, stream));
    for (int r = 0; r < kReplays; ++r)
        TORCH_CUDA_CHECK(cudaGraphLaunch(exec, stream));
    TORCH_CUDA_CHECK(cudaEventRecord(stop, stream));
    TORCH_CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    TORCH_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    return static_cast<double>(ms) * 1000.0 / (static_cast<double>(kReplays) * launches);
}

// Highest GPU clock seen in sysfs while the object lives, in MHz (as in bench/memory_bandwidth.cu).
class ClockSampler {
public:
    ClockSampler() {
        thread_ = std::thread([this] {
            while (!stop_) {
                std::ifstream f("/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq");
                long hz = 0;
                if (f >> hz)
                    peak_ = std::max<long>(peak_, hz / 1000000);
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        });
    }
    ~ClockSampler() {
        stop_ = true;
        thread_.join();
    }
    long peak_mhz() const { return peak_; }

private:
    std::atomic<bool> stop_{false};
    std::atomic<long> peak_{0};
    std::thread thread_;
};

std::string command_output(const char* command) {
    std::string out;
    if (std::FILE* p = popen(command, "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, p))
            out += line;
        pclose(p);
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == ' '))
        out.pop_back();
    return out.empty() ? "unknown" : out;
}

// ---- The cases ----------------------------------------------------------------------------------------------------

// One timed kernel shape and how often a token launches it.
struct Row {
    std::string kernel, name, format, shape;
    double bytes = 0;         // moved per launch, see the header comment
    double weight_bytes = 0;  // of which weights
    double us = 0;            // per launch
    int count = 0;            // launches per token

    double gbs() const { return bytes / (us * 1e-6) / 1e9; }
    double ms_per_token() const { return count * us / 1000.0; }
};

// Columns of a verify window (--columns): the dense matrix products run on this many inputs at once; bytes count the
// weights once and the inputs and outputs per column.
int g_columns = 1;

// A matrix-vector product the forward pass runs.
struct Matvec {
    std::string kernel;  // gemv_q8, gemv_bf16, gemv_q8_group or gemv_q8_experts
    std::string name;    // the tensors that share the shape, e.g. "attn_gate/ssm_out"
    Type type;
    int rows, cols;
    bool fused = false;  // experts and groups: gate and up in one launch, with their SwiGLU
    int count = 0;
    int rows2 = 0;  // BF16: rows of a second matrix in the same launch (a mixer's inject rows, or a pair)
    std::vector<std::pair<Type, int>> parts;  // gemv_q8_group: each matrix's format and rows
    bool quantizes = true;                    // quantized products: the launch's input is quantized for it (not reused)
    float silu_scale = 0.0f;                  // BF16: the first matrix's rows store silu(silu_scale * sum)
    int collapse_hc = 0;                      // BF16: a mixer's up projection run with hc_collapse over hc streams

    std::string format() const;
    std::string shape() const;
};

std::string format_name(Type t) {
    return std::string(type_info(static_cast<uint32_t>(t))->name);
}

std::string shape_name(int rows, int cols) {
    return std::to_string(rows) + "x" + std::to_string(cols);
}

std::string Matvec::format() const {
    if (kernel == "mixer_down_q8" && rows2 > 0)
        return format_name(type) + "+BF16";
    std::string f;
    for (const auto& [t, r] : parts)
        f += (f.empty() ? "" : "+") + format_name(t);
    return parts.empty() ? format_name(type) : f;
}

std::string Matvec::shape() const {
    std::string r = std::to_string(rows) + (rows2 > 0 ? "+" + std::to_string(rows2) : "");
    if (!parts.empty()) {
        r.clear();
        for (const auto& [t, n] : parts)
            r += (r.empty() ? "" : "+") + std::to_string(n);
    }
    return r + "x" + std::to_string(cols);
}

// The "blk.N." prefix and ".weight" suffix removed; empty for tensors that are not weights.
std::string short_name(const std::string& tensor) {
    std::string n = tensor;
    if (n.rfind("blk.", 0) == 0)
        n = n.substr(n.find('.', 4) + 1);
    const std::string suffix = ".weight";
    if (n.size() <= suffix.size() || n.compare(n.size() - suffix.size(), suffix.size(), suffix) != 0)
        return "";
    return n.substr(0, n.size() - suffix.size());
}

// The deduplicated matrix-vector products of one token with their counts, from the tensor table.
std::vector<Matvec> matvec_cases(const gguf::File& file, const model::Config& cfg, int& swiglu_launches,
                                 int& silu_launches) {
    static const char* const kMatrices[] = {"attn_qkv",
                                            "attn_gate",
                                            "ssm_out",
                                            "ssm_alpha",
                                            "ssm_beta",
                                            "attn_q",
                                            "attn_k",
                                            "attn_v",
                                            "attn_output",
                                            "ffn_gate_inp",
                                            "ffn_gate_inp_shexp",
                                            "ffn_gate_shexp",
                                            "ffn_up_shexp",
                                            "ffn_down_shexp",
                                            "hc_attn_down",
                                            "hc_attn_up",
                                            "hc_attn_inject",
                                            "hc_ffn_down",
                                            "hc_ffn_up",
                                            "hc_ffn_inject",
                                            "output_hc_down",
                                            "output_hc_up",
                                            "output",
                                            "ple_key",
                                            "ple_value"};
    std::vector<Matvec> cases;
    std::map<std::tuple<std::string, std::string, std::string, bool, bool>, size_t> index;
    auto add_case = [&](const Matvec& m) {
        const auto key = std::make_tuple(m.kernel, m.format(), m.shape(), m.fused, m.quantizes);
        auto [at, fresh] = index.try_emplace(key, cases.size());
        if (fresh)
            cases.push_back(m);
        Matvec& c = cases[at->second];
        if (!fresh && c.name.find(m.name) == std::string::npos)
            c.name += "/" + m.name;
        ++c.count;
    };
    auto add = [&](const std::string& kernel, const std::string& name, Type type, int rows, int cols, bool fused,
                   int rows2 = 0) { add_case({kernel, name, type, rows, cols, fused, 0, rows2}); };

    // Matrices that read one input share launches as model.cpp runs them (products, experts): the quantized ones one
    // gemv_q8_group launch when it takes their formats (the shared expert's gate and up with their SwiGLU, on the input
    // the routed experts quantized), the BF16 ones in pairs.
    static const std::vector<std::vector<std::string>> kShared = {{"attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha"},
                                                                  {"attn_q", "attn_k", "attn_v"},
                                                                  {"ffn_gate_inp", "ffn_gate_inp_shexp"},
                                                                  {"ffn_gate_shexp", "ffn_up_shexp"}};
    for (int il = 0; il < cfg.n_layer; ++il) {
        for (const std::vector<std::string>& names : kShared) {
            std::vector<const gguf::TensorInfo*> bf16, q8;
            for (const std::string& n : names)
                if (const gguf::TensorInfo* t = file.tensor("blk." + std::to_string(il) + "." + n + ".weight"))
                    (model::matrix_type(*t) == Type::BF16 ? bf16 : q8).push_back(t);
            const bool swiglu = names[0] == "ffn_gate_shexp";
            const bool grouped = q8.size() > 1 && std::all_of(q8.begin(), q8.end(), [](const gguf::TensorInfo* t) {
                                     return kernels::gemv_q8_groups(static_cast<Type>(t->type));
                                 });
            // Their input, the block's mixed vector, arrives quantized from hc_collapse.
            Matvec group{"gemv_q8_group", "", Type::F32, 0, 0, swiglu};
            group.quantizes = false;
            for (size_t i = 0; i < q8.size(); ++i) {
                const gguf::TensorInfo& t = *q8[i];
                const Type type = static_cast<Type>(t.type);
                const int rows = static_cast<int>(t.shape[1]), cols = static_cast<int>(t.shape[0]);
                if (!kernels::gemv_q8_supports(type))
                    throw Error("Tensor " + t.name + " is " + format_name(type) + ", which no kernel reads.");
                if (grouped) {
                    group.name += (group.name.empty() ? "" : "+") + short_name(t.name);
                    group.parts.emplace_back(type, rows);
                    group.cols = cols;
                    group.rows += rows;
                } else {
                    Matvec m{"gemv_q8", short_name(t.name), type, rows, cols, false};
                    m.quantizes = false;
                    add_case(m);
                }
            }
            if (grouped)
                add_case(group);
            if (swiglu && !grouped)
                ++swiglu_launches;
            for (size_t i = 0; i < bf16.size(); i += 2) {
                const gguf::TensorInfo& t = *bf16[i];
                const int rows = static_cast<int>(t.shape[1]), cols = static_cast<int>(t.shape[0]);
                if (i + 1 < bf16.size())
                    add("gemv_bf16", short_name(t.name) + "+" + short_name(bf16[i + 1]->name), Type::BF16, rows, cols,
                        false, static_cast<int>(bf16[i + 1]->shape[1]));
                else
                    add("gemv_bf16", short_name(t.name), Type::BF16, rows, cols, false);
            }
        }
    }
    auto shared_input = [&](const std::string& name) {
        for (const std::vector<std::string>& names : kShared)
            if (std::find(names.begin(), names.end(), name) != names.end())
                return true;
        return false;
    };

    // A mixer's BF16 inject rows run in its BF16 down projection's launch (model.cpp, hc_mix).
    auto inject_of = [&](const gguf::TensorInfo& down) -> const gguf::TensorInfo* {
        const size_t at = down.name.find("hc_");
        const size_t end = down.name.rfind("_down.weight");
        if (at == std::string::npos || end == std::string::npos ||
            (model::matrix_type(down) != Type::BF16 && model::matrix_type(down) != Type::Q8_0))
            return nullptr;
        const gguf::TensorInfo* inject = file.tensor(down.name.substr(0, end) + "_inject.weight");
        return inject != nullptr && model::matrix_type(*inject) == Type::BF16 && inject->shape[0] == down.shape[0]
                   ? inject
                   : nullptr;
    };
    for (const gguf::TensorInfo& t : file.tensors()) {
        const std::string name = short_name(t.name);
        const Type type = model::matrix_type(t);
        const int rows = static_cast<int>(t.shape[1]), cols = static_cast<int>(t.shape[0]);
        if (t.name.rfind("blk.", 0) == 0 && shared_input(name))
            continue;  // above
        if (std::find(std::begin(kMatrices), std::end(kMatrices), name) != std::end(kMatrices)) {
            const size_t inject_at = t.name.rfind("_inject.weight");
            if (inject_at != std::string::npos) {
                const gguf::TensorInfo* down = file.tensor(t.name.substr(0, inject_at) + "_down.weight");
                if (down != nullptr && inject_of(*down) == &t)
                    continue;  // timed with its down projection
            }
            // A mixer's BF16 down projection stores its low-rank SiLU, and its up projection runs with hc_collapse
            // (hc_mix).
            const bool mixer_down = name.find("hc_") != std::string::npos && name.find("_down") != std::string::npos;
            const bool mixer_up = name.find("hc_") != std::string::npos && name.size() > 3 &&
                                  name.compare(name.size() - 3, 3, "_up") == 0;
            if (type == Type::Q8_0 && mixer_down) {
                Matvec m{"mixer_down_q8", name, type, rows, cols, false};
                if (const auto* inject = inject_of(t)) {
                    m.name += "+inject";
                    m.rows2 = static_cast<int>(inject->shape[1]);
                }
                m.silu_scale = 1.0f / static_cast<float>(cfg.hc);
                add_case(m);
            } else if (type == Type::Q8_0 && mixer_up && cfg.hc == 4 && cols == 320 && rows % (32 * cfg.hc) == 0) {
                Matvec m{"mixer_up_q8", name, type, rows, cols, false};
                m.collapse_hc = cfg.hc;
                m.quantizes = false;
                add_case(m);
            } else if (type == Type::BF16 && mixer_up && cfg.hc == 4 && cols % 8 == 0 && cols <= 320 &&
                       rows % (32 * cfg.hc) == 0) {
                Matvec m{"gemv_bf16_collapse", name, type, rows, cols, false};
                m.collapse_hc = cfg.hc;
                add_case(m);
            } else if (type == Type::BF16) {
                Matvec m{"gemv_bf16", name, type, rows, cols, false};
                if (const gguf::TensorInfo* inject = inject_of(t)) {
                    m.name += "+inject";
                    m.rows2 = static_cast<int>(inject->shape[1]);
                }
                m.silu_scale = mixer_down ? 1.0f / static_cast<float>(cfg.hc) : 0.0f;
                add_case(m);
            } else if (kernels::gemv_q8_supports(type)) {
                if (mixer_down)
                    ++silu_launches;  // a quantized down projection runs silu_scale on its own
                // The head reads the mixed vector and ssm_out the gated norm's output, both written quantized.
                Matvec m{"gemv_q8", name, type, rows, cols, false};
                m.quantizes = name != "output" && name != "ssm_out";
                add_case(m);
            } else
                throw Error("Tensor " + t.name + " is " + format_name(type) + ", which no matrix-vector kernel reads.");
        } else if (name == "ffn_gate_exps") {
            const std::string up = t.name.substr(0, t.name.find("ffn_gate_exps")) + "ffn_up_exps.weight";
            const gguf::TensorInfo* u = file.tensor(up);
            if (u == nullptr || u->type != t.type || !kernels::gemv_q8_supports(type))
                throw Error("Layer tensors " + t.name + " and " + up + " must exist in one supported format.");
            add("gemv_q8_experts", "gate+up exps", type, rows, cols, true);
        } else if (name == "ffn_down_exps") {
            if (!kernels::gemv_q8_supports(type))
                throw Error("Tensor " + t.name + " is " + format_name(type) + ", which no expert kernel reads.");
            add("gemv_q8_experts", "down exps", type, rows, cols, false);
        }
    }
    return cases;
}

// A plain matrix: copies of the weights rotated through a graph, one launch per copy.
Row time_matrix(const Matvec& c, cudaStream_t stream, Rng& rng) {
    const uint64_t row_size = row_bytes(*type_info(static_cast<uint32_t>(c.type)), c.cols);
    const bool q8_down = c.kernel == "mixer_down_q8";
    const uint64_t first_bytes = static_cast<uint64_t>(c.rows) * row_size;
    const uint64_t second_bytes = static_cast<uint64_t>(c.rows2) * (q8_down ? 2ull * c.cols : row_size);
    const uint64_t matrix_bytes = first_bytes + second_bytes;
    const uint64_t stride = round_up(matrix_bytes, 256);
    const uint64_t copies = copies_for(matrix_bytes);
    Pinned weights(stride * copies);
    if (q8_down && second_bytes) {
        fill_weights(c.type, weights.host(), first_bytes, rng);
        fill_weights(Type::BF16, weights.host() + first_bytes, second_bytes, rng);
    } else
        fill_weights(c.type, weights.host(), matrix_bytes, rng);
    for (uint64_t i = 1; i < copies; ++i)
        std::memcpy(weights.host() + i * stride, weights.host(), matrix_bytes);
    const int launches = static_cast<int>(std::max<uint64_t>(kMinLaunches, copies));
    const int T = g_columns;
    Device<float> y(static_cast<size_t>(c.rows + c.rows2) * T);

    Row row{c.kernel, c.name, c.format(), c.shape()};
    row.weight_bytes = static_cast<double>(matrix_bytes);
    row.count = c.count;
    if (c.collapse_hc > 0) {
        const int n = c.rows / c.collapse_hc;
        Device<float> x = random_floats(static_cast<size_t>(c.cols) * T, -1.0f, 1.0f, rng),
                      xn = random_floats(static_cast<size_t>(c.rows) * T, -1.0f, 1.0f, rng),
                      mixed(static_cast<size_t>(n) * T);
        Device<BlockQ8> q8(static_cast<size_t>(n / 32) * T);
        row.bytes = row.weight_bytes + T * (4.0 * (c.cols + c.rows + n) + n / 32.0 * sizeof(BlockQ8));
        row.us = time_graph(stream, launches, [&](int i) {
            const uint8_t* w = weights.device() + stride * (static_cast<uint64_t>(i) % copies);
            if (c.type == Type::Q8_0)
                kernels::mixer_up_q8(w, n, x.get(), xn.get(), mixed.get(), q8.get(), T, stream);
            else
                kernels::gemv_bf16_collapse(w, n, c.collapse_hc, c.cols, x.get(), xn.get(), mixed.get(), q8.get(),
                                            stream, T);
        });
    } else if (q8_down) {
        Device<float> x = random_floats(static_cast<size_t>(c.cols) * T, -1.0f, 1.0f, rng);
        Device<BlockQ8> xq(static_cast<size_t>(c.cols / 32) * T);
        kernels::quantize_q8(x.get(), int64_t{c.cols} * T, xq.get(), stream);
        row.bytes = row.weight_bytes +
                    T * (c.cols / 32.0 * sizeof(BlockQ8) + 4.0 * (c.rows + c.rows2 + (c.rows2 > 0 ? c.cols : 0)));
        row.us = time_graph(stream, launches, [&](int i) {
            const uint8_t* w = weights.device() + stride * (static_cast<uint64_t>(i) % copies);
            kernels::mixer_down_q8(w, c.rows2 > 0 ? w + first_bytes : nullptr, c.rows, c.rows2, c.cols, xq.get(),
                                   x.get(), y.get(), y.get() + static_cast<size_t>(c.rows) * T, c.silu_scale, T,
                                   stream);
        });
    } else if (c.type == Type::BF16) {
        Device<float> x = random_floats(static_cast<size_t>(c.cols) * T, -1.0f, 1.0f, rng);
        row.bytes = row.weight_bytes + 4.0 * T * (c.cols + c.rows + c.rows2);
        row.us = time_graph(stream, launches, [&](int i) {
            const uint8_t* w = weights.device() + stride * (static_cast<uint64_t>(i) % copies);
            kernels::gemv_bf16_pair(w, c.rows, w + c.rows * row_size, c.rows2, c.cols, x.get(), y.get(),
                                    y.get() + static_cast<size_t>(c.rows) * T, stream, c.silu_scale, T);
        });
    } else {
        const size_t blocks = static_cast<size_t>(c.cols) / 32;
        Device<BlockQ8> x = random_q8(blocks * T, rng);
        row.bytes = row.weight_bytes + T * (static_cast<double>(blocks * sizeof(BlockQ8)) + 4.0 * c.rows);
        row.us = time_graph(stream, launches, [&](int i) {
            kernels::gemv_q8(c.type, weights.device() + stride * (static_cast<uint64_t>(i) % copies), c.rows, c.cols,
                             x.get(), y.get(), stream, T);
        });
    }
    return row;
}

// Dense matrices with one input in one gemv_q8_group launch, rotated through copies like time_matrix.
Row time_group(const Matvec& c, cudaStream_t stream, Rng& rng) {
    std::vector<uint64_t> offsets;
    uint64_t group_bytes = 0;
    for (const auto& [type, rows] : c.parts) {
        offsets.push_back(group_bytes);
        group_bytes +=
            round_up(static_cast<uint64_t>(rows) * row_bytes(*type_info(static_cast<uint32_t>(type)), c.cols), 256);
    }
    const uint64_t copies = copies_for(group_bytes);
    Pinned weights(group_bytes * copies);
    double weight_bytes = 0;
    for (size_t p = 0; p < c.parts.size(); ++p) {
        const auto [type, rows] = c.parts[p];
        const uint64_t bytes = static_cast<uint64_t>(rows) * row_bytes(*type_info(static_cast<uint32_t>(type)), c.cols);
        fill_weights(type, weights.host() + offsets[p], bytes, rng);
        weight_bytes += static_cast<double>(bytes);
    }
    for (uint64_t i = 1; i < copies; ++i)
        std::memcpy(weights.host() + i * group_bytes, weights.host(), group_bytes);
    const int launches = static_cast<int>(std::max<uint64_t>(kMinLaunches, copies));
    const size_t blocks = static_cast<size_t>(c.cols) / 32;
    const int T = g_columns;
    Device<BlockQ8> x = random_q8(blocks * T, rng);
    Device<float> y(static_cast<size_t>(c.rows) * T);

    Row row{c.kernel, c.name, c.format(), c.shape()};
    row.weight_bytes = weight_bytes;
    row.bytes = weight_bytes +
                T * (static_cast<double>(blocks * sizeof(BlockQ8)) + 4.0 * (c.fused ? c.parts[0].second : c.rows));
    row.count = c.count;
    row.us = time_graph(stream, launches, [&](int i) {
        kernels::Q8Part parts[3];
        size_t at = 0;
        for (size_t p = 0; p < c.parts.size(); ++p) {
            parts[p] = {c.parts[p].first,
                        weights.device() + group_bytes * (static_cast<uint64_t>(i) % copies) + offsets[p],
                        c.parts[p].second, y.get() + at};
            at += static_cast<size_t>(c.parts[p].second) * T;
        }
        kernels::gemv_q8_group(parts, static_cast<int>(c.parts.size()), c.cols, x.get(), c.fused, stream, T);
    });
    return row;
}

// Routed experts: `slots` distinct experts per launch from a pool of slices, through pointer tables.
Row time_experts(const Matvec& c, int slots, cudaStream_t stream, Rng& rng) {
    const uint64_t slice = static_cast<uint64_t>(c.rows) * row_bytes(*type_info(static_cast<uint32_t>(c.type)), c.cols);
    const uint64_t parts = c.fused ? 2 : 1;  // gate and up
    const uint64_t pool_slices = std::max<uint64_t>(static_cast<uint64_t>(slots), copies_for(slice * parts));
    const int launches = static_cast<int>(std::max<uint64_t>(kMinLaunches, (pool_slices + slots - 1) / slots));

    std::vector<std::unique_ptr<Pinned>> pools, tables;
    for (uint64_t p = 0; p < parts; ++p) {
        pools.push_back(std::make_unique<Pinned>(slice * pool_slices));
        fill_weights(c.type, pools.back()->host(), slice, rng);
        for (uint64_t s = 1; s < pool_slices; ++s)
            std::memcpy(pools.back()->host() + s * slice, pools.back()->host(), slice);
        tables.push_back(std::make_unique<Pinned>(sizeof(uint8_t*) * static_cast<size_t>(launches * slots)));
        auto* entries = reinterpret_cast<const uint8_t**>(tables.back()->host());
        for (int i = 0; i < launches; ++i)
            for (int s = 0; s < slots; ++s)
                entries[i * slots + s] =
                    pools.back()->device() + slice * ((static_cast<uint64_t>(i) * slots + s) % pool_slices);
    }

    const size_t blocks_per_slot = static_cast<size_t>(c.cols) / 32;
    const size_t x_blocks = c.fused ? blocks_per_slot : blocks_per_slot * static_cast<size_t>(slots);
    const int64_t x_stride = c.fused ? 0 : static_cast<int64_t>(blocks_per_slot);
    Device<BlockQ8> x = random_q8(x_blocks, rng);
    Device<float> y(static_cast<size_t>(slots) * c.rows);
    auto table = [&](size_t part, int i) {
        return reinterpret_cast<const uint8_t* const*>(tables[part]->device()) + static_cast<size_t>(i) * slots;
    };

    Row row{c.kernel, c.name + " x" + std::to_string(slots), format_name(c.type), shape_name(c.rows, c.cols)};
    row.weight_bytes = static_cast<double>(slice * parts * static_cast<uint64_t>(slots));
    row.bytes = row.weight_bytes + static_cast<double>(x_blocks * sizeof(BlockQ8)) + 4.0 * slots * c.rows;
    row.count = c.count;
    row.us = time_graph(stream, launches, [&](int i) {
        kernels::gemv_q8_experts(c.type, table(0, i), c.fused ? table(1, i) : nullptr, slots, c.rows, c.cols, x.get(),
                                 x_stride, y.get(), stream);
    });
    return row;
}

// The routed gate and up (gemv_q8_experts_window) or down (gemv_q8_experts_down_window, c.fused false) of a window of
// T = g_columns columns: each column routes `k` distinct
// experts: each of the previous column's experts with probability 1/2, the rest drawn uniformly from `experts`
// (about 25 unique experts of 40 routes at T = 4; review measured 26). Deduplicated as the window gate does; the row
// reports the mean unique count. Bytes count each unique expert's matrices once.
Row time_experts_window(const Matvec& c, int k, int experts, cudaStream_t stream, Rng& rng) {
    constexpr int kSlots = 80;
    const int T = g_columns;
    const uint64_t slice = static_cast<uint64_t>(c.rows) * row_bytes(*type_info(static_cast<uint32_t>(c.type)), c.cols);
    const uint64_t pool_slices = std::max<uint64_t>(kSlots, copies_for(slice * (c.fused ? 2 : 1)));
    const int launches = static_cast<int>(std::max<uint64_t>(kMinLaunches, (pool_slices + kSlots - 1) / kSlots));
    std::vector<std::unique_ptr<Pinned>> pools;
    const int parts = c.fused ? 2 : 1;
    for (int p = 0; p < parts; ++p) {
        pools.push_back(std::make_unique<Pinned>(slice * pool_slices));
        fill_weights(c.type, pools.back()->host(), slice, rng);
        for (uint64_t s = 1; s < pool_slices; ++s)
            std::memcpy(pools.back()->host() + s * slice, pools.back()->host(), slice);
    }
    // per launch: gate[80], up[80] pointers, then columns[80], rank[80 * 8], count
    struct Tables {
        const uint8_t* gate[kSlots];
        const uint8_t* up[kSlots];
        uint8_t columns[kSlots];
        int8_t rank[kSlots * kernels::kMaxColumns];
        int32_t count;
    };
    Pinned tables(sizeof(Tables) * static_cast<size_t>(launches));
    auto* host = reinterpret_cast<Tables*>(tables.host());
    double unique = 0;
    for (int i = 0; i < launches; ++i) {
        Tables& t = host[i];
        std::memset(&t, 0, sizeof t);
        std::memset(t.rank, -1, sizeof t.rank);
        std::vector<int> seen, ids_prev;
        for (int col = 0; col < T; ++col) {
            std::vector<int> routes;
            for (int r = 0; col > 0 && r < k; ++r) {
                const int e = ids_prev[static_cast<size_t>(r)];
                if (rng.next() % 2 == 0)
                    routes.push_back(e);
            }
            while (static_cast<int>(routes.size()) < k) {
                const int e = static_cast<int>(rng.next() % static_cast<uint64_t>(experts));
                if (std::find(routes.begin(), routes.end(), e) == routes.end())
                    routes.push_back(e);
            }
            ids_prev = routes;
            for (int r = 0; r < k; ++r) {
                auto it = std::find(seen.begin(), seen.end(), routes[r]);
                const int u = static_cast<int>(it - seen.begin());
                if (it == seen.end()) {
                    seen.push_back(routes[r]);
                    const uint64_t at = slice * ((static_cast<uint64_t>(i) * kSlots + u) % pool_slices);
                    t.gate[u] = pools[0]->device() + at;
                    t.up[u] = pools[static_cast<size_t>(parts - 1)]->device() + at;
                }
                t.columns[u] |= static_cast<uint8_t>(1u << col);
                t.rank[u * kernels::kMaxColumns + col] = static_cast<int8_t>(r);
            }
        }
        t.count = static_cast<int32_t>(seen.size());
        unique += static_cast<double>(seen.size()) / launches;
    }
    const size_t units = static_cast<size_t>(c.cols) / 32;
    const size_t inputs = c.fused ? static_cast<size_t>(T) : static_cast<size_t>(T) * k;  // down: one a slot
    Device<BlockQ8> x = random_q8(units * inputs, rng);
    Device<float> h(static_cast<size_t>(T) * k * c.rows);
    const auto* dev = reinterpret_cast<const Tables*>(tables.device());

    char name[96];
    std::snprintf(name, sizeof name, "%s x%d, %.1f unique", c.name.c_str(), k, unique);
    Row row{c.fused ? "gemv_q8_experts_window" : "gemv_q8_experts_down_window", name, format_name(c.type),
            shape_name(c.rows, c.cols)};
    row.weight_bytes = static_cast<double>(slice) * parts * unique;
    row.bytes = row.weight_bytes + static_cast<double>(units * inputs * sizeof(BlockQ8)) + 4.0 * T * k * c.rows;
    row.count = c.count;
    row.us = time_graph(stream, launches, [&](int i) {
        const Tables& t = dev[i];
        if (c.fused)
            kernels::gemv_q8_experts_window(c.type, t.gate, t.up, t.columns, t.rank, &t.count, std::min(T * k, kSlots),
                                            k, c.rows, c.cols, x.get(), h.get(), T, stream);
        else
            kernels::gemv_q8_experts_down_window(c.type, t.gate, t.columns, t.rank, &t.count, std::min(T * k, kSlots),
                                                 k, c.rows, c.cols, x.get(), h.get(), T, stream);
    });
    return row;
}

// ---- The small kernels --------------------------------------------------------------------------------------------

// Times the kernels between the matrices, once per token shape. `quantize_counts` maps a vector length to the
// number of quantize_q8 launches a token makes on it; `swiglu_launches` counts the shared experts whose gate and up
// do not share a launch, `silu_launches` the mixers whose down projection does not apply the SiLU itself and
// `collapse_launches` those whose up projection does not run hc_collapse.
void time_small_kernels(const model::Config& cfg, const std::map<int, int>& quantize_counts, int swiglu_launches,
                        int silu_launches, int collapse_launches, cudaStream_t stream, Rng& rng,
                        std::vector<Row>& rows) {
    const int E = cfg.n_embd, HC = cfg.hc, LR = cfg.hc_rank, D = cfg.head_dim, H = cfg.n_head, HKV = cfg.n_head_kv;
    const int DS = cfg.ssm_head_dim, HV = cfg.ssm_v_heads, KH = cfg.ssm_k_heads, CC = cfg.conv_channels();
    const int K = cfg.n_expert_used, NE = cfg.n_expert, FFS = cfg.n_ff_shared;
    const int HCD = HC * E, KV = HKV * D, QK = KH * DS;
    int recurrent = 0;
    for (int il = 0; il < cfg.n_layer; ++il)
        recurrent += cfg.recurrent(il);
    const int attention = cfg.n_layer - recurrent, layers = cfg.n_layer;
    constexpr int kLaunches = 64;
    constexpr double F = sizeof(float);

    auto add = [&](const char* kernel, const std::string& name, const std::string& shape, double bytes, int count,
                   int launches, auto&& launch) {
        Row r{kernel, name, "F32", shape};
        r.bytes = bytes;
        r.count = count;
        r.us = time_graph(stream, launches, launch);
        rows.push_back(std::move(r));
    };
    auto floats = [&](size_t n, float lo = -1.0f, float hi = 1.0f) { return random_floats(n, lo, hi, rng); };

    // quantize_q8: every matvec of a quantized weight quantizes its input again, as model.cpp does.
    for (auto [n, count] : quantize_counts) {
        Device<float> x = floats(static_cast<size_t>(n));
        Device<BlockQ8> out(static_cast<size_t>(n) / 32);
        add("quantize_q8", "n=" + std::to_string(n), std::to_string(n), n * F + n / 32.0 * sizeof(BlockQ8), count,
            kLaunches, [&](int) { kernels::quantize_q8(x.get(), n, out.get(), stream); });
    }

    // Norms.
    {
        Device<float> x = floats(static_cast<size_t>(HCD)), w = floats(static_cast<size_t>(HCD), 0.5f, 1.5f),
                      out(static_cast<size_t>(HCD));
        // 97 hyper-connection mixers (two per layer and the head): all but the first layer's attention mixer and the
        // one after the PLE block start with hc_combine_norm (below); those two and the 3 norms of the PLE block are
        // here.
        add("rms_norm", "hc mixer, 4 x 2560 + weight", shape_name(HC, E), 3.0 * HCD * F,
            (cfg.ple_layer > 0 ? 2 : 1) + 3, kLaunches,
            [&](int) { kernels::rms_norm(x.get(), E, w.get(), E, nullptr, 0, out.get(), E, E, HC, cfg.eps, stream); });
    }
    {
        Device<float> o = floats(static_cast<size_t>(HV * DS)), z = floats(static_cast<size_t>(HV * DS)),
                      w = floats(static_cast<size_t>(DS), 0.5f, 1.5f);
        Device<kernels::BlockQ8> o_q8(static_cast<size_t>(HV * DS) / 32);
        add("rms_norm_q8", "gated (in place), Q8 output", shape_name(HV, DS),
            (3.0 * HV * DS + DS) * F + HV * DS / 32.0 * sizeof(kernels::BlockQ8), recurrent, kLaunches, [&](int) {
                kernels::rms_norm_q8(o.get(), DS, w.get(), 0, z.get(), DS, o.get(), DS, o_q8.get(), DS, HV, cfg.eps,
                                     stream);
            });
    }

    // Hyper-connection helpers.
    {
        Device<float> lo = floats(static_cast<size_t>(LR));
        if (silu_launches > 0)
            add("silu_scale", "n=" + std::to_string(LR), std::to_string(LR), 2.0 * LR * F, silu_launches, kLaunches,
                [&](int) { kernels::silu_scale(lo.get(), LR, 1.0f / static_cast<float>(HC), stream); });
        Device<float> xn = floats(static_cast<size_t>(HCD)), gate = floats(static_cast<size_t>(HCD)),
                      mixed(static_cast<size_t>(E));
        Device<kernels::BlockQ8> mixed_q8(static_cast<size_t>(E) / 32);
        add("hc_collapse", shape_name(HC, E) + " with Q8 output", shape_name(HC, E),
            (2.0 * HCD + E) * F + E / 32.0 * sizeof(kernels::BlockQ8), collapse_launches, kLaunches,
            [&](int) { kernels::hc_collapse(xn.get(), gate.get(), mixed.get(), mixed_q8.get(), E, HC, stream); });
        Device<float> res = floats(static_cast<size_t>(HCD)), out = floats(static_cast<size_t>(E)),
                      inject = floats(static_cast<size_t>(HC));
        // Each block's output is added to the residual in the next mixer's norm launch, except before the PLE block.
        const int separate = cfg.ple_layer > 0 ? 1 : 0;
        add("hc_combine", shape_name(HC, E), shape_name(HC, E), (2.0 * HCD + E + HC) * F, separate, kLaunches,
            [&](int) { kernels::hc_combine(res.get(), out.get(), inject.get(), E, HC, stream); });
        Device<float> w = floats(static_cast<size_t>(HCD), 0.5f, 1.5f), xn2(static_cast<size_t>(HCD));
        add("hc_combine_norm", shape_name(HC, E), shape_name(HC, E), (4.0 * HCD + E + HC) * F, 2 * layers - separate,
            kLaunches, [&](int) {
                kernels::hc_combine_norm(res.get(), out.get(), inject.get(), w.get(), xn2.get(), E, HC, cfg.eps,
                                         stream);
            });
        add("hc_broadcast", shape_name(HC, E), shape_name(HC, E), (HCD + E) * F, 1, kLaunches,
            [&](int) { kernels::hc_broadcast(out.get(), res.get(), E, HC, stream); });
    }

    // Gated DeltaNet.
    {
        Device<float> state = floats(static_cast<size_t>(3 * CC)), x = floats(static_cast<size_t>(CC)),
                      w = floats(static_cast<size_t>(4 * CC)), out(static_cast<size_t>(CC));
        add("conv_silu_l2", "channels=" + std::to_string(CC), std::to_string(CC),
            (2.0 * 3 * CC + CC + 4.0 * CC + CC) * F, recurrent, kLaunches,
            [&](int) { kernels::conv_silu_l2(state.get(), x.get(), w.get(), out.get(), CC, 2 * QK, cfg.eps, stream); });
        // gdn_step stores the gates back in place, so repeated launches gate them again: harmless for timing.
        Device<float> beta = floats(static_cast<size_t>(HV)), alpha = floats(static_cast<size_t>(HV)),
                      dt = floats(static_cast<size_t>(HV)), a = floats(static_cast<size_t>(HV), -1.0f, -0.01f);

        // The state is read and written once per layer; rotate through copies so L2 never holds it.
        const size_t state_floats = static_cast<size_t>(HV) * DS * DS;
        const uint64_t copies = copies_for(state_floats * sizeof(float));
        Device<float> states(state_floats * copies);
        TORCH_CUDA_CHECK(cudaMemset(states.get(), 0, states.count() * sizeof(float)));
        Device<float> qkv = floats(static_cast<size_t>(CC)), o(static_cast<size_t>(HV * DS));
        const float scale = 1.0f / std::sqrt(static_cast<float>(DS));
        add("gdn_step", "heads=" + std::to_string(HV), shape_name(HV, DS),
            2.0 * state_floats * F + (2.0 * QK + 2.0 * HV * DS + 6.0 * HV) * F, recurrent, static_cast<int>(copies),
            [&](int i) {
                kernels::gdn_step(states.get() + state_floats * (static_cast<uint64_t>(i) % copies), qkv.get(),
                                  qkv.get() + QK, qkv.get() + 2 * QK, alpha.get(), beta.get(), dt.get(), a.get(),
                                  o.get(), KH, HV, scale, stream);
            });
    }

    // Gated DeltaNet in a window of T columns (count 0: not part of a token; divide the time by T for a column).
    {
        constexpr int kT = 8;
        Device<float> state = floats(static_cast<size_t>(3 * CC)), x = floats(static_cast<size_t>(kT * CC)),
                      w = floats(static_cast<size_t>(4 * CC)), out(static_cast<size_t>(kT * CC));
        Device<float> beta = floats(static_cast<size_t>(kT * HV)), alpha = floats(static_cast<size_t>(kT * HV)),
                      dt = floats(static_cast<size_t>(HV)), a = floats(static_cast<size_t>(HV), -1.0f, -0.01f),
                      gates(static_cast<size_t>(kT * HV * 2));
        const size_t state_floats = static_cast<size_t>(HV) * DS * DS;
        const uint64_t copies = std::max<uint64_t>(copies_for(state_floats * sizeof(float)), 2);
        Device<float> states(state_floats * copies), o(static_cast<size_t>(kT * HV * DS));
        TORCH_CUDA_CHECK(cudaMemset(states.get(), 0, states.count() * sizeof(float)));
        const float scale = 1.0f / std::sqrt(static_cast<float>(DS));
        for (int t : {2, 4, 8}) {
            const std::string at_t = ", T=" + std::to_string(t);
            add("conv_silu_l2_window", "channels=" + std::to_string(CC) + at_t, std::to_string(CC),
                (3.0 * CC + t * (CC + 4.0 * CC + CC)) * F, 0, kLaunches, [&](int) {
                    kernels::conv_silu_l2_window(state.get(), x.get(), w.get(), out.get(), CC, 2 * QK, cfg.eps, t,
                                                 stream);
                });
            add("conv_commit", "channels=" + std::to_string(CC) + at_t, std::to_string(CC), 6.0 * CC * F, 0, kLaunches,
                [&](int) { kernels::conv_commit(state.get(), x.get(), CC, t, stream); });
            add("gdn_window", "heads=" + std::to_string(HV) + at_t, shape_name(HV, DS),
                2.0 * state_floats * F + t * (2.0 * QK + 2.0 * HV * DS + 6.0 * HV) * F, 0, static_cast<int>(copies),
                [&](int i) {
                    const size_t from = state_floats * (static_cast<uint64_t>(i) % copies);
                    const size_t to = state_floats * (static_cast<uint64_t>(i + 1) % copies);
                    kernels::gdn_window(states.get() + from, states.get() + to, out.get(), out.get() + QK,
                                        out.get() + 2 * QK, CC, alpha.get(), beta.get(), dt.get(), a.get(), gates.get(),
                                        o.get(), KH, HV, scale, t, stream);
                });
        }
    }

    // Full attention. The cache rotates through copies, one per layer's worth of bytes.
    {
        const size_t cache_elements = static_cast<size_t>(kAttentionContext) * KV;
        const uint64_t copies = copies_for(2 * cache_elements * sizeof(uint16_t));
        std::vector<uint16_t> row(cache_elements);
        for (uint16_t& h : row)
            h = static_cast<uint16_t>((rng.next() & 0x83FF) | 0x3000);  // finite F16 around +-0.2
        Device<uint16_t> k_cache(cache_elements * copies), v_cache(cache_elements * copies);
        for (uint64_t c = 0; c < copies; ++c)
            k_cache.upload(row, c * cache_elements), v_cache.upload(row, c * cache_elements);
        Device<int32_t> pos(1);
        pos.upload({kAttentionPosition});
        constexpr int kT = 8;  // the buffers hold a full window of columns; T = 1 uses the first one
        Device<float> q = floats(static_cast<size_t>(kT * 2 * H * D)), kk = floats(static_cast<size_t>(kT * KV)),
                      vv = floats(static_cast<size_t>(kT * KV)), out(static_cast<size_t>(kT * H * D));
        Device<float> scratch(static_cast<size_t>(kernels::attention_scratch_floats(H, kAttentionContext, kT)));
        const float scale = 1.0f / std::sqrt(static_cast<float>(D));

        Device<float> wq = floats(static_cast<size_t>(D), 0.5f, 1.5f), wk = floats(static_cast<size_t>(D), 0.5f, 1.5f);
        add("qk_norm_rope_store", "q and k heads", shape_name(H + HKV, D),
            (2.0 * H * D + 2.0 * KV + KV + 2.0 * D) * F + 2.0 * KV * sizeof(uint16_t), attention, kLaunches, [&](int) {
                kernels::qk_norm_rope_store(q.get(), 2 * D, wq.get(), kk.get(), wk.get(), vv.get(), k_cache.get(),
                                            v_cache.get(), H, HKV, cfg.n_rot, pos.get(), cfg.rope_base, cfg.eps, 1,
                                            stream);
            });
        add("attention_decode", "position " + std::to_string(kAttentionPosition), shape_name(H, D),
            2.0 * (kAttentionPosition + 1) * KV * sizeof(uint16_t) + (2.0 * H * D + H * D) * F, attention,
            static_cast<int>(copies), [&](int i) {
                const size_t at = cache_elements * (static_cast<uint64_t>(i) % copies);
                kernels::attention_decode(q.get(), q.get() + D, 2 * D, k_cache.get() + at, v_cache.get() + at,
                                          out.get(), scratch.get(), pos.get(), H, HKV, scale, kAttentionContext, 1,
                                          stream);
            });
        // Verify windows of T columns at the same position: the K and V rows are read once per window. These rows
        // are not part of a token (count 0); divide the time by T for the cost per column.
        for (int t : {2, 4, 8}) {
            const std::string at_t = ", T=" + std::to_string(t);
            add("qk_norm_rope_store", "q and k heads" + at_t, shape_name(H + HKV, D),
                t * ((2.0 * H * D + 2.0 * KV + KV + 2.0 * D) * F + 2.0 * KV * sizeof(uint16_t)), 0, kLaunches,
                [&](int) {
                    kernels::qk_norm_rope_store(q.get(), 2 * D, wq.get(), kk.get(), wk.get(), vv.get(), k_cache.get(),
                                                v_cache.get(), H, HKV, cfg.n_rot, pos.get(), cfg.rope_base, cfg.eps, t,
                                                stream);
                });
            add("attention_decode", "position " + std::to_string(kAttentionPosition) + at_t, shape_name(H, D),
                2.0 * (kAttentionPosition + t) * KV * sizeof(uint16_t) + t * (2.0 * H * D + H * D) * F, 0,
                static_cast<int>(copies), [&](int i) {
                    const size_t at = cache_elements * (static_cast<uint64_t>(i) % copies);
                    kernels::attention_decode(q.get(), q.get() + D, 2 * D, k_cache.get() + at, v_cache.get() + at,
                                              out.get(), scratch.get(), pos.get(), H, HKV, scale, kAttentionContext, t,
                                              stream);
                });
        }
    }

    // Experts' glue and the shared expert's activation.
    {
        Device<float> logits = floats(static_cast<size_t>(NE)), weights(static_cast<size_t>(K));
        Device<int32_t> ids(static_cast<size_t>(K));
        add("router_top_k", "n=" + std::to_string(NE), std::to_string(NE), (NE + 2.0 * K) * F, layers, kLaunches,
            [&](int) { kernels::router_top_k(logits.get(), NE, K, ids.get(), weights.get(), stream); });
        Device<float> g = floats(static_cast<size_t>(FFS)), u = floats(static_cast<size_t>(FFS)),
                      h(static_cast<size_t>(FFS));
        if (swiglu_launches > 0)
            add("swiglu", "n=" + std::to_string(FFS), std::to_string(FFS), 3.0 * FFS * F, swiglu_launches, kLaunches,
                [&](int) { kernels::swiglu(g.get(), u.get(), h.get(), FFS, stream); });
        Device<float> y = floats(static_cast<size_t>(K * E)), w = floats(static_cast<size_t>(K), 0.0f, 0.2f),
                      ys = floats(static_cast<size_t>(E)), sg = floats(1), out(static_cast<size_t>(E));
        add("moe_combine", shape_name(K, E), shape_name(K, E), (static_cast<double>(K) * E + K + 2.0 * E + 1) * F,
            layers, kLaunches,
            [&](int) { kernels::moe_combine(y.get(), w.get(), K, ys.get(), sg.get(), out.get(), E, stream); });
    }

    // The per-layer embedding block, in one layer.
    {
        Device<float> key_n = floats(static_cast<size_t>(HCD)), query = floats(static_cast<size_t>(HCD)),
                      gate(static_cast<size_t>(HC));
        add("ple_gate", shape_name(HC, E), shape_name(HC, E), (2.0 * HCD + HC) * F, 1, kLaunches,
            [&](int) { kernels::ple_gate(key_n.get(), query.get(), gate.get(), E, HC, stream); });
        Device<float> value = floats(static_cast<size_t>(E)), gated(static_cast<size_t>(HCD));
        add("ple_gated_value", shape_name(HC, E), shape_name(HC, E), (E + HC + static_cast<double>(HCD)) * F, 1,
            kLaunches, [&](int) { kernels::ple_gated_value(value.get(), gate.get(), gated.get(), E, HC, stream); });
        Device<float> res = floats(static_cast<size_t>(HCD)), x = floats(static_cast<size_t>(HCD)),
                      hist = floats(static_cast<size_t>(9 * HCD)), w = floats(static_cast<size_t>(4 * HCD));
        Device<int32_t> pos(1);
        pos.upload({kAttentionPosition});
        // res read and written, gated, x, three history rows read and one written, four taps per channel.
        add("ple_conv", "channels=" + std::to_string(HCD), std::to_string(HCD), 12.0 * HCD * F, 1, kLaunches, [&](int) {
            kernels::ple_conv(res.get(), gated.get(), x.get(), hist.get(), w.get(), HCD, pos.get(), stream);
        });
        Device<float> hist_w = floats(static_cast<size_t>(9 * HCD));
        Device<float> res_w = floats(static_cast<size_t>(8 * HCD)), gated_w = floats(static_cast<size_t>(8 * HCD)),
                      x_w = floats(static_cast<size_t>(8 * HCD));
        for (int t : {2, 4, 8}) {
            const std::string at_t = ", T=" + std::to_string(t);
            add("ple_conv_window", "channels=" + std::to_string(HCD) + at_t, std::to_string(HCD),
                (t * 12.0 - 2.0 * t) * HCD * F, 0, kLaunches, [&](int) {
                    kernels::ple_conv_window(res_w.get(), gated_w.get(), x_w.get(), hist_w.get(), w.get(), HCD,
                                             pos.get(), t, stream);
                });
            add("ple_commit", "channels=" + std::to_string(HCD) + at_t, std::to_string(HCD), 2.0 * t * HCD * F, 0,
                kLaunches, [&](int) { kernels::ple_commit(hist_w.get(), x_w.get(), HCD, pos.get(), t, stream); });
        }
    }
}

// ---- Output -------------------------------------------------------------------------------------------------------

struct Totals {
    double ms = 0, bytes = 0, weight_bytes = 0, small_ms = 0;
    int launches = 0, small_launches = 0;
};

Totals totals_of(const std::vector<Row>& rows) {
    Totals t;
    for (const Row& r : rows) {
        t.ms += r.ms_per_token();
        t.bytes += r.bytes * r.count;
        t.weight_bytes += r.weight_bytes * r.count;
        t.launches += r.count;
        if (r.bytes < static_cast<double>(kSmallBytes))
            t.small_ms += r.ms_per_token(), t.small_launches += r.count;
    }
    return t;
}

void print_table(const std::vector<Row>& rows, const Totals& t, double ceiling) {
    std::printf("\n%-16s %-36s %-8s %-11s %9s %9s %8s %6s %5s %10s\n", "kernel", "case", "format", "shape", "MB/launch",
                "us/launch", "GB/s", "% ceil", "count", "ms/token");
    for (const Row& r : rows) {
        std::printf("%-16s %-36.36s %-8s %-11s %9.3f %9.2f %8.1f %5.0f%% %5d %10.3f\n", r.kernel.c_str(),
                    r.name.c_str(), r.format.c_str(), r.shape.c_str(), r.bytes / 1e6, r.us, r.gbs(),
                    100.0 * r.gbs() / ceiling, r.count, r.ms_per_token());
    }
    std::printf("\nsummed kernel time per token   %9.3f ms\n", t.ms);
    std::printf("bytes per token                %9.3f GB (weights %.3f GB)\n", t.bytes / 1e9, t.weight_bytes / 1e9);
    std::printf("kernel-only decode rate        %9.2f tok/s\n", 1000.0 / t.ms);
    std::printf("mean bandwidth                 %9.1f GB/s (%.0f%% of the %.0f GB/s ceiling)\n",
                t.bytes / (t.ms * 1e-3) / 1e9, 100.0 * t.bytes / (t.ms * 1e-3) / 1e9 / ceiling, ceiling);
    std::printf("launches per token             %9d\n", t.launches);
    std::printf("launches under 1 MiB           %9d, %.3f ms (%.1f%% of kernel time)\n", t.small_launches, t.small_ms,
                100.0 * t.small_ms / t.ms);
}

std::string json_string(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\')
            out += '\\';
        out += (c == '\n') ? ' ' : c;
    }
    return out + "\"";
}

void write_json(const char* path, const std::vector<Row>& rows, const Totals& t, double ceiling, long peak_mhz,
                const std::string& model, const std::string& power_mode) {
    std::FILE* f = std::fopen(path, "w");
    if (f == nullptr)
        throw Error(std::string("Cannot write ") + path + "; choose a writable path for --json.");
    char date[64];
    const std::time_t now = std::time(nullptr);
    std::strftime(date, sizeof date, "%Y-%m-%dT%H:%M:%S%z", std::localtime(&now));
    std::fprintf(f, "{\n  \"date\": \"%s\",\n  \"commit\": %s,\n  \"model\": %s,\n  \"power_mode\": %s,\n", date,
                 json_string(TORCH_GIT_SHA).c_str(),
                 json_string(std::filesystem::path(model).filename().string()).c_str(),
                 json_string(power_mode).c_str());
    std::fprintf(f, "  \"gpu_clock_peak_mhz\": %ld,\n  \"ceiling_gbs\": %.1f,\n  \"columns\": %d,\n  \"cases\": [\n",
                 peak_mhz, ceiling, g_columns);
    for (size_t i = 0; i < rows.size(); ++i) {
        const Row& r = rows[i];
        std::fprintf(f,
                     "    {\"kernel\": %s, \"case\": %s, \"format\": %s, \"shape\": %s, \"bytes\": %.0f, "
                     "\"weight_bytes\": %.0f, \"us\": %.3f, \"gbs\": %.2f, \"percent_of_ceiling\": %.1f, "
                     "\"count_per_token\": %d, \"ms_per_token\": %.5f}%s\n",
                     json_string(r.kernel).c_str(), json_string(r.name).c_str(), json_string(r.format).c_str(),
                     json_string(r.shape).c_str(), r.bytes, r.weight_bytes, r.us, r.gbs(), 100.0 * r.gbs() / ceiling,
                     r.count, r.ms_per_token(), i + 1 < rows.size() ? "," : "");
    }
    std::fprintf(f,
                 "  ],\n  \"totals\": {\"kernel_ms_per_token\": %.4f, \"bytes_per_token\": %.0f, "
                 "\"weight_bytes_per_token\": %.0f, \"tokens_per_second\": %.2f, \"launches_per_token\": %d, "
                 "\"small_launches_per_token\": %d, \"small_launch_ms_per_token\": %.4f}\n}\n",
                 t.ms, t.bytes, t.weight_bytes, 1000.0 / t.ms, t.launches, t.small_launches, t.small_ms);
    std::fclose(f);
}

int run(const std::string& model, const char* json_path, double ceiling) {
    const gguf::File file = gguf::File::open_model(model, /*header_only=*/true);
    const model::Config cfg = model::Config::read(file);
    int swiglu_launches = 0;  // the shared experts' SwiGLU, where their gate and up are not one group launch
    int silu_launches = 0;    // silu_scale, where a mixer's down projection is not BF16
    const std::vector<Matvec> cases = matvec_cases(file, cfg, swiglu_launches, silu_launches);

    cudaStream_t stream = nullptr;
    TORCH_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    kernels::upload_iq_tables();
    int sms = 0;
    TORCH_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const std::string power_mode = command_output("nvpmodel -q 2>/dev/null | head -1");
    std::printf("torch_bench_kernels %s, %d SMs, power mode: %s, ceiling %.0f GB/s\n", TORCH_GIT_SHA, sms,
                power_mode.c_str(), ceiling);
    ClockSampler clock;

    Rng rng;
    std::vector<Row> rows;
    std::map<int, int> quantize_counts;
    for (const Matvec& c : cases) {
        std::fprintf(stderr, "  %-16s %-34s %-8s %s x%d\n", c.kernel.c_str(), c.name.c_str(), c.format().c_str(),
                     c.shape().c_str(), c.count);
        if (c.kernel == "gemv_q8_experts") {
            // a window's gate and up run on its unique experts, its down on every (column, rank)
            const int T = torch::bench::g_columns;
            rows.push_back(T > 1 ? time_experts_window(c, cfg.n_expert_used, cfg.n_expert, stream, rng)
                                 : time_experts(c, cfg.n_expert_used, stream, rng));
            // Gate and up read the mixed input as hc_collapse quantized it; the activated experts are quantized for
            // down.
            if (!c.fused)
                quantize_counts[cfg.n_expert_used * torch::bench::g_columns * c.cols] += c.count;
        } else {
            rows.push_back(c.kernel == "gemv_q8_group" ? time_group(c, stream, rng) : time_matrix(c, stream, rng));
            if (c.type != Type::BF16 && c.quantizes)
                quantize_counts[c.cols] += c.count;
        }
    }
    int collapse_launches = 2 * cfg.n_layer + 1;  // the mixers whose up projection does not run hc_collapse itself
    for (const Matvec& c : cases)
        collapse_launches -= c.collapse_hc > 0 ? c.count : 0;
    time_small_kernels(cfg, quantize_counts, swiglu_launches, silu_launches, collapse_launches, stream, rng, rows);
    TORCH_CUDA_CHECK(cudaStreamDestroy(stream));

    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.ms_per_token() > b.ms_per_token(); });
    const Totals totals = totals_of(rows);
    print_table(rows, totals, ceiling);
    std::printf("highest GPU clock seen: %ld MHz\n", clock.peak_mhz());
    if (json_path != nullptr)
        write_json(json_path, rows, totals, ceiling, clock.peak_mhz(), model, power_mode);
    return 0;
}

}  // namespace
}  // namespace torch::bench

int main(int argc, char** argv) {
    std::string model;
    const char* json_path = nullptr;
    double ceiling = 180.0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--json" && i + 1 < argc)
            json_path = argv[++i];
        else if (arg == "--ceiling" && i + 1 < argc)
            ceiling = std::atof(argv[++i]);
        else if (arg == "--columns" && i + 1 < argc)
            torch::bench::g_columns = std::atoi(argv[++i]);
        else if (arg.rfind("--", 0) != 0 && model.empty())
            model = arg;
        else
            model.clear(), ceiling = 0;
    }
    if (model.empty() || ceiling <= 0 || torch::bench::g_columns < 1 || torch::bench::g_columns > 8) {
        std::fprintf(stderr,
                     "usage: torch_bench_kernels MODEL-00001-of-0000N.gguf [--json FILE] [--ceiling GB/s] "
                     "[--columns 1-8]\n");
        return 2;
    }
    try {
        return torch::bench::run(model, json_path, ceiling);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "torch_bench_kernels: %s\n", e.what());
        return 1;
    }
}
