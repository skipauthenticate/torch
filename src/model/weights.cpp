#include "model/weights.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/error.h"
#include "formats/dequant.h"
#include "kernels/gemv/gemv.h"

namespace torch::model {

// Binds tensors to the fields of Weights while laying out the arena. In the dry run (no arena) it only adds up
// the bytes; with an arena it also records the page-aligned reads that fill it.
class WeightLoader {
public:
    WeightLoader(const gguf::File& f, uint8_t* arena, int n_expert) : f_(f), arena_(arena), n_expert_(n_expert) {}

    uint64_t used() const noexcept { return used_; }

    struct Read {
        uint64_t file_offset;  // page-aligned
        uint64_t bytes;        // from file_offset to the tensor's last byte (DirectFile reads the whole pages)
        uint8_t* dst;          // page-aligned
    };
    std::vector<Read> reads;

    struct Convert {  // an F16 vector read into the arena, converted to F32 after the reads
        const uint8_t* f16;
        float* f32;
        int64_t n;
    };
    std::vector<Convert> converts;

    // An F32 matrix narrowed to BF16 in place after the reads: the values must be BF16 values, so nothing changes.
    struct Narrow {
        std::string name;
        uint8_t* data;  // n F32 values, then n BF16 values from the same address
        int64_t n;
    };
    std::vector<Narrow> narrows;

    // A BF16 or quantized matrix as stored, or an F32 one as BF16 (narrowed after the reads).
    Matrix matrix(const std::string& name, int64_t cols, int64_t rows) {
        const gguf::TensorInfo& t = tensor(name, {cols, rows});
        const Type type = matrix_type(t);
        if (type == Type::BF16 ? cols % 8 != 0 : !kernels::gemv_q8_supports(type))
            throw Error("Tensor " + name + " is stored as " + std::string(type_info(t.type)->name) +
                        ", which the matrix-vector kernels do not read; use the supported model file.");
        const uint8_t* data = place(t, t.nbytes);
        if (t.type == static_cast<uint32_t>(Type::F32) && arena_)
            narrows.push_back({name, const_cast<uint8_t*>(data), t.elements()});
        return {type, static_cast<int>(rows), static_cast<int>(cols), data};
    }

    const float* vector(const std::string& name, std::initializer_list<int64_t> shape) {
        const gguf::TensorInfo& t = tensor(name, shape);
        const int64_t n = t.elements();
        const Type type = static_cast<Type>(t.type);
        if (type == Type::F32)
            return reinterpret_cast<const float*>(place(t, t.nbytes));
        if (type != Type::F16)
            throw Error("Tensor " + name + " is " + std::string(type_info(t.type)->name) +
                        ", but the engine expects F32 or F16; use the supported model file.");
        const uint8_t* raw = place(t, t.nbytes);
        float* out = reinterpret_cast<float*>(room(static_cast<uint64_t>(n) * sizeof(float)));
        if (arena_)
            converts.push_back({raw, out, n});
        return out;
    }

    // A [cols, rows, n_expert] tensor: its layout only; runtime::ExpertStore reads the experts.
    ExpertTensor experts(const std::string& name, int64_t cols, int64_t rows) {
        const gguf::TensorInfo& t = tensor(name, {cols, rows, n_expert_});
        const Type type = static_cast<Type>(t.type);
        if (!kernels::gemv_q8_supports(type))
            throw Error("Expert tensor " + name + " is " + std::string(type_info(t.type)->name) +
                        ", which the expert kernels do not read; use the supported model file.");
        return {type, static_cast<int>(rows), static_cast<int>(cols), t.slice_bytes(), f_.data_offset() + t.offset};
    }

    // Every expert of an expert tensor experts() checked, in the arena (the draft layer's are all resident).
    const uint8_t* all_experts(const std::string& name) {
        const gguf::TensorInfo& t = *f_.tensor(name);
        return place(t, t.nbytes);
    }

private:
    const gguf::TensorInfo& tensor(const std::string& name, std::initializer_list<int64_t> shape) {
        const gguf::TensorInfo* t = f_.tensor(name);
        if (t == nullptr)
            throw Error("The model file has no tensor " + name +
                        "; use the Qwen3.8-Flash-Next GGUF this engine "
                        "supports.");
        size_t d = 0;
        for (int64_t want : shape) {
            if (t->shape[d++] != want)
                throw Error("Tensor " + name + " has an unexpected shape; use the supported model file.");
        }
        for (; d < 4; ++d) {
            if (t->shape[d] != 1)
                throw Error("Tensor " + name + " has an unexpected shape; use the supported model file.");
        }
        return *t;
    }

    // The first n bytes of tensor t, at the same offset within a page as in the file, so O_DIRECT can read them in.
    const uint8_t* place(const gguf::TensorInfo& t, uint64_t n) {
        const uint64_t offset = f_.data_offset() + t.offset;
        const uint64_t span = runtime::span_bytes(offset, n);
        uint8_t* page = room(span);
        if (arena_ == nullptr)
            return nullptr;
        reads.push_back({runtime::page_floor(offset), offset + n - runtime::page_floor(offset), page});
        return page + (offset - runtime::page_floor(offset));
    }

    uint8_t* room(uint64_t bytes) {
        uint8_t* at = arena_ ? arena_ + used_ : nullptr;
        used_ += runtime::page_ceil(bytes);
        return at;
    }

    const gguf::File& f_;
    uint8_t* arena_;
    int64_t n_expert_;
    uint64_t used_ = 0;
};

namespace {

// The tensors of one layer, named with prefix b ("blk.N."), its kind already set in L (recurrent or attention).
void bind_layer(WeightLoader& w, const Config& c, const std::string& b, Layer& L) {
    const int64_t E = c.n_embd, HCD = int64_t{c.hc} * E, LR = c.hc_rank;
    const int64_t QK = int64_t{c.ssm_k_heads} * c.ssm_head_dim, V = int64_t{c.ssm_v_heads} * c.ssm_head_dim;
    const int64_t D = c.head_dim, KV = int64_t{c.n_head_kv} * D;
    L.hc_attn_norm = w.vector(b + "hc_attn_norm.weight", {HCD});
    L.hc_attn_down = w.matrix(b + "hc_attn_down.weight", HCD, LR);
    L.hc_attn_up = w.matrix(b + "hc_attn_up.weight", LR, HCD);
    L.hc_attn_inject = w.matrix(b + "hc_attn_inject.weight", HCD, c.hc);
    L.hc_ffn_norm = w.vector(b + "hc_ffn_norm.weight", {HCD});
    L.hc_ffn_down = w.matrix(b + "hc_ffn_down.weight", HCD, LR);
    L.hc_ffn_up = w.matrix(b + "hc_ffn_up.weight", LR, HCD);
    L.hc_ffn_inject = w.matrix(b + "hc_ffn_inject.weight", HCD, c.hc);

    if (L.recurrent) {
        L.qkv = w.matrix(b + "attn_qkv.weight", E, 2 * QK + V);
        L.z = w.matrix(b + "attn_gate.weight", E, V);
        L.beta = w.matrix(b + "ssm_beta.weight", E, c.ssm_v_heads);
        L.alpha = w.matrix(b + "ssm_alpha.weight", E, c.ssm_v_heads);
        L.conv = w.vector(b + "ssm_conv1d.weight", {c.ssm_conv, c.conv_channels()});
        L.dt_bias = w.vector(b + "ssm_dt.bias", {c.ssm_v_heads});
        L.a = w.vector(b + "ssm_a", {c.ssm_v_heads});
        L.ssm_norm = w.vector(b + "ssm_norm.weight", {c.ssm_head_dim});
        L.ssm_out = w.matrix(b + "ssm_out.weight", V, E);
    } else {
        L.wq = w.matrix(b + "attn_q.weight", E, 2 * D * c.n_head);
        L.wk = w.matrix(b + "attn_k.weight", E, KV);
        L.wv = w.matrix(b + "attn_v.weight", E, KV);
        L.wo = w.matrix(b + "attn_output.weight", D * c.n_head, E);
        L.q_norm = w.vector(b + "attn_q_norm.weight", {D});
        L.k_norm = w.vector(b + "attn_k_norm.weight", {D});
    }

    L.router = w.matrix(b + "ffn_gate_inp.weight", E, c.n_expert);
    L.shared_gate_inp = w.matrix(b + "ffn_gate_inp_shexp.weight", E, 1);
    L.shared_gate = w.matrix(b + "ffn_gate_shexp.weight", E, c.n_ff_shared);
    L.shared_up = w.matrix(b + "ffn_up_shexp.weight", E, c.n_ff_shared);
    L.shared_down = w.matrix(b + "ffn_down_shexp.weight", c.n_ff_shared, E);
    L.gate_exps = w.experts(b + "ffn_gate_exps.weight", E, c.n_ff_expert);
    L.up_exps = w.experts(b + "ffn_up_exps.weight", E, c.n_ff_expert);
    L.down_exps = w.experts(b + "ffn_down_exps.weight", c.n_ff_expert, E);
    if (L.gate_exps.type != L.up_exps.type)
        throw Error("The tensors " + b +
                    "ffn_gate_exps and ffn_up_exps are stored in different formats; use the "
                    "supported model file.");
}

// Lays out an arena for the tensors `bind` binds, allocates it as mapped pinned memory and reads them into it.
template <class Bind>
uint8_t* load_arena(const gguf::File& f, const runtime::DirectFile& file, runtime::ReadPool& pool, int n_expert,
                    uint64_t& bytes, Bind&& bind) {
    WeightLoader dry(f, nullptr, n_expert);
    bind(dry);
    bytes = dry.used();
    void* arena = nullptr;
    TORCH_CUDA_CHECK(cudaHostAlloc(&arena, bytes, cudaHostAllocMapped));
    // The kernels take arena addresses as they are; that needs the GPU to see mapped memory at the host address.
    void* device = nullptr;
    TORCH_CUDA_CHECK(cudaHostGetDevicePointer(&device, arena, 0));
    if (device != arena) {
        cudaFreeHost(arena);
        throw Error(
            "The GPU maps pinned memory at a different address than the CPU; the engine needs unified "
            "addressing (a Jetson Orin with JetPack 7).");
    }

    WeightLoader loader(f, static_cast<uint8_t*>(arena), n_expert);
    bind(loader);

    // Large tensors (the output head, 0.5 GiB) are split so the pool keeps several reads in flight.
    constexpr uint64_t kChunk = 64 * (uint64_t{1} << 20);
    std::vector<WeightLoader::Read> chunks;
    for (const WeightLoader::Read& r : loader.reads) {
        for (uint64_t at = 0; at < r.bytes; at += kChunk)
            chunks.push_back({r.file_offset + at, std::min(kChunk, r.bytes - at), r.dst + at});
    }
    try {
        pool.run(static_cast<int>(chunks.size()), [&](int i) {
            const WeightLoader::Read& r = chunks[static_cast<size_t>(i)];
            file.read(r.file_offset, r.bytes, r.dst);
        });
        for (const WeightLoader::Convert& c : loader.converts) {
            for (int64_t i = 0; i < c.n; ++i)
                c.f32[i] = fp16_to_f32(load_u16(c.f16 + 2 * i));
        }
        // Forward in place: BF16 value i (bytes 2i, 2i + 1) never overwrites an F32 value not yet read.
        for (const WeightLoader::Narrow& m : loader.narrows) {
            for (int64_t i = 0; i < m.n; ++i) {
                uint32_t v;
                std::memcpy(&v, m.data + 4 * i, 4);
                if ((v & 0xffffu) != 0)
                    throw Error("Tensor " + m.name +
                                " is stored as F32 with values BF16 cannot hold exactly; this engine reads such "
                                "tensors only as BF16, so use a supported model file (docs/model-files.md).");
                const auto top = static_cast<uint16_t>(v >> 16);
                std::memcpy(m.data + 2 * i, &top, 2);
            }
        }
    } catch (...) {
        cudaFreeHost(arena);
        throw;
    }
    return static_cast<uint8_t*>(arena);
}

}  // namespace

void Weights::bind(WeightLoader& w, const Config& c) {
    const int64_t E = c.n_embd, HCD = int64_t{c.hc} * E, LR = c.hc_rank;

    token_embd = w.matrix("token_embd.weight", E, c.n_vocab);
    head_hc_norm = w.vector("output_hc_norm.weight", {HCD});
    head_hc_down = w.matrix("output_hc_down.weight", HCD, LR);
    head_hc_up = w.matrix("output_hc_up.weight", LR, HCD);
    output = w.matrix("output.weight", E, c.n_vocab);

    layers.resize(static_cast<size_t>(c.n_layer));
    for (int il = 0; il < c.n_layer; ++il) {
        Layer& L = layers[static_cast<size_t>(il)];
        const std::string b = "blk." + std::to_string(il) + ".";
        L.recurrent = c.recurrent(il);
        L.ple = il == c.ple_layer;

        bind_layer(w, c, b, L);
        if (!L.recurrent) {
            L.idx_q = w.matrix(b + "indexer.q_proj.weight", E, 4 * 128);
            L.idx_k = w.matrix(b + "indexer.k_proj.weight", E, 128);
            L.idx_q_norm = w.vector(b + "indexer.q_norm.weight", {128});
            L.idx_k_norm = w.vector(b + "indexer.k_norm.weight", {128});
        }
        if (L.ple) {
            L.ple_key = w.matrix(b + "ple_key.weight", E, HCD);
            L.ple_value = w.matrix(b + "ple_value.weight", E, E);
            L.ple_norm_key = w.vector(b + "ple_norm_key.weight", {HCD});
            L.ple_norm_query = w.vector(b + "ple_norm_query.weight", {HCD});
            L.ple_norm_conv = w.vector(b + "ple_norm_conv.weight", {HCD});
            L.ple_conv = w.vector(b + "ple_conv1d.weight", {c.ple_conv, HCD});
        }
    }
}

uint64_t Weights::base_bytes(const Config& config, const gguf::File& shard1) {
    WeightLoader dry(shard1, nullptr, config.n_expert);
    Weights w;
    w.bind(dry, config);
    return dry.used();
}

Weights::Weights(const Config& config, const gguf::File& shard1, const runtime::DirectFile& file,
                 runtime::ReadPool& pool) {
    arena_ = load_arena(shard1, file, pool, config.n_expert, arena_bytes_, [&](WeightLoader& w) { bind(w, config); });
}

Weights::~Weights() {
    if (arena_)
        cudaFreeHost(arena_);
}

void DraftWeights::bind(WeightLoader& w, const Config& c) {
    const int64_t E = c.n_embd, HCD = int64_t{c.hc} * E, LR = c.hc_rank;
    const std::string b = "blk." + std::to_string(c.n_layer) + ".", n = b + "nextn.";
    layer = Layer{};  // an attention layer: recurrent and ple stay false
    bind_layer(w, c, b, layer);
    gate_exps = w.all_experts(b + "ffn_gate_exps.weight");
    up_exps = w.all_experts(b + "ffn_up_exps.weight");
    down_exps = w.all_experts(b + "ffn_down_exps.weight");
    eh_proj = w.matrix(n + "eh_proj.weight", 2 * E, E);
    enorm = w.vector(n + "enorm.weight", {E});
    hnorm = w.vector(n + "hnorm.weight", {HCD});
    head_hc_norm = w.vector(n + "hc_head_norm.weight", {HCD});
    head_hc_down = w.matrix(n + "hc_head_down.weight", HCD, LR);
    head_hc_up = w.matrix(n + "hc_head_up.weight", LR, HCD);
}

uint64_t DraftWeights::bytes(const Config& config, const gguf::File& draft) {
    WeightLoader dry(draft, nullptr, config.n_expert);
    DraftWeights w;
    w.bind(dry, config);
    return dry.used();
}

DraftWeights::DraftWeights(const Config& config, const gguf::File& draft, const runtime::DirectFile& file,
                           runtime::ReadPool& pool) {
    if (draft.get<uint32_t>("qwen4exp.nextn_predict_layers") != 1u)
        throw Error(
            "The draft file has no qwen4exp.nextn_predict_layers = 1, so it is not a draft-layer file; build "
            "one with tools/make_draft.py.");
    arena_ = load_arena(draft, file, pool, config.n_expert, arena_bytes_, [&](WeightLoader& w) { bind(w, config); });
    // eh_proj's rows are [fc_embedding row | fc_hidden row]: split them into two matrices, in place.
    const uint64_t half = torch::row_bytes(*type_info(static_cast<uint32_t>(eh_proj.type)), config.n_embd);
    if (2 * half != eh_proj.row_bytes())
        throw Error("The draft layer's eh_proj cannot be split at column " + std::to_string(config.n_embd) +
                    " in its format; build the file with tools/make_draft.py.");
    uint8_t* data = const_cast<uint8_t*>(eh_proj.data);
    const std::vector<uint8_t> rows(data, data + eh_proj.row_bytes() * static_cast<uint64_t>(eh_proj.rows));
    for (uint64_t r = 0; r < static_cast<uint64_t>(eh_proj.rows); ++r) {
        std::memcpy(data + r * half, rows.data() + 2 * r * half, half);
        std::memcpy(data + (eh_proj.rows + r) * half, rows.data() + (2 * r + 1) * half, half);
    }
    fc_embedding = {eh_proj.type, eh_proj.rows, config.n_embd, data};
    fc_hidden = {eh_proj.type, eh_proj.rows, config.n_embd, data + eh_proj.rows * half};
}

DraftWeights::~DraftWeights() {
    if (arena_)
        cudaFreeHost(arena_);
}

}  // namespace torch::model
