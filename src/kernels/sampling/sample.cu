#include "core/error.h"
#include "kernels/common.cuh"
#include "kernels/sampling/sample.h"

namespace torch::kernels {
namespace {

constexpr int kThreads = 1024;
constexpr uint32_t kPurposeSample = 1;

// ---- Philox-4x32-10 --------------------------------------------------------------------------------------------

__device__ __forceinline__ void mulhilo(uint32_t a, uint32_t b, uint32_t& hi, uint32_t& lo) {
    const uint64_t product = static_cast<uint64_t>(a) * b;
    hi = static_cast<uint32_t>(product >> 32);
    lo = static_cast<uint32_t>(product);
}

// The first of the four output words for counter (c0, c1, 0, 0) and key = seed, as a float in [0, 1).
__device__ float uniform(uint64_t seed, uint32_t c0, uint32_t c1) {
    uint32_t c[4] = {c0, c1, 0u, 0u};
    uint32_t k0 = static_cast<uint32_t>(seed), k1 = static_cast<uint32_t>(seed >> 32);
    for (int round = 0; round < 10; ++round) {
        uint32_t hi0, lo0, hi1, lo1;
        mulhilo(0xD2511F53u, c[0], hi0, lo0);
        mulhilo(0xCD9E8D57u, c[2], hi1, lo1);
        const uint32_t next[4] = {hi1 ^ c[1] ^ k0, lo1, hi0 ^ c[3] ^ k1, lo0};
        c[0] = next[0], c[1] = next[1], c[2] = next[2], c[3] = next[3];
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return static_cast<float>(c[0] >> 8) * (1.0f / 16777216.0f);
}

// ---- Block helpers ---------------------------------------------------------------------------------------------

// p * 2^40 as an integer. The float product is exact (a power of two), and double-precision arithmetic is slow on this
// GPU, so none is used here.
__device__ __forceinline__ unsigned long long fixed_mass(float p) {
    return __float2ull_rn(p * 1099511627776.0f);
}

constexpr int kStripe = 256;  // contiguous indices per stripe; a vocabulary of 248,320 has 970 stripes
constexpr int kStripes = kThreads;
constexpr int kLanes = kWarp;

struct Shared {
    unsigned int count[kLanes][65];  // radix histograms, one copy per lane so no two lanes of a warp share an address
    unsigned long long mass[kLanes][65];  // (65 wide so the lanes of a warp also fall in different banks)
    unsigned int count_total[64];
    unsigned long long mass_total[64];
    unsigned long long stripe_weight[kStripes];  // per stripe, then the exclusive prefixes over stripes
    unsigned long long stripe_prefix[kStripes];
    unsigned long long tile[kStripe];
    unsigned long long reduce[kThreads / kWarp];
    float best_value[kThreads / kWarp];
    int best_index[kThreads / kWarp];
    unsigned int prefix, chosen;
    unsigned long long remaining, total_mass, total_weight;
    unsigned int ties, remaining_count;
    int owner, found;
};

// A selection boundary in the order (p descending, index ascending): tokens with p above `bits` are in, and so are
// tokens whose p equals `bits` and whose index is at most `last`. active == false keeps everything.
struct Boundary {
    uint32_t bits;
    int last;
    bool active;
    __device__ bool keeps(float p, int v) const {
        if (!active)
            return true;
        const uint32_t key = __float_as_uint(p);
        return key > bits || (key == bits && v <= last);
    }
};

__device__ __forceinline__ unsigned long long warp_sum_u64(unsigned long long v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

// Exact sum over the block of one 64-bit integer per thread (any order gives the same bits).
__device__ unsigned long long block_sum_u64(Shared& sh, unsigned long long v) {
    v = warp_sum_u64(v);
    __syncthreads();
    if (threadIdx.x % kWarp == 0)
        sh.reduce[threadIdx.x / kWarp] = v;
    __syncthreads();
    unsigned long long total = 0;
    for (int w = 0; w < kThreads / kWarp; ++w)
        total += sh.reduce[w];
    return total;
}

// Per stripe of kStripe indices, the sum of weight(probs[v], v) (a 64-bit integer, read with the warp's lanes on
// consecutive indices), then the exclusive prefix over stripes; returns the total. Integers add exactly in any order.
template <class Weight>
__device__ unsigned long long scan_stripes(Shared& sh, const float* probs, int vocab, Weight weight) {
    const int tid = threadIdx.x, lane = tid % kWarp, warp = tid / kWarp;
    const int stripes = (vocab + kStripe - 1) / kStripe;
    for (int stripe = warp; stripe < kStripes; stripe += kThreads / kWarp) {
        unsigned long long sum = 0;
        if (stripe < stripes)
#pragma unroll
            for (int i = 0; i < kStripe / (kWarp * 4); ++i) {
                const int v = stripe * kStripe + i * kWarp * 4 + lane * 4;
                if (v < vocab) {
                    const float4 p = *reinterpret_cast<const float4*>(probs + v);
                    sum += weight(p.x, v) + weight(p.y, v + 1) + weight(p.z, v + 2) + weight(p.w, v + 3);
                }
            }
        sum = warp_sum_u64(sum);
        if (lane == 0)
            sh.stripe_weight[stripe] = sum;
    }
    __syncthreads();
    // Hillis-Steele inclusive scan, then shift to exclusive.
    unsigned long long value = sh.stripe_weight[tid];
    for (int offset = 1; offset < kStripes; offset <<= 1) {
        sh.stripe_prefix[tid] = value;
        __syncthreads();
        if (tid >= offset)
            value += sh.stripe_prefix[tid - offset];
        __syncthreads();
    }
    sh.stripe_prefix[tid] = value - sh.stripe_weight[tid];
    if (tid == kStripes - 1)
        sh.total_weight = value;
    __syncthreads();
    return sh.total_weight;
}

// The index v whose cumulative weight (in increasing index order, after scan_stripes with the same `weight`) first
// exceeds `target` < total.
template <class Weight>
__device__ int locate(Shared& sh, const float* probs, int vocab, Weight weight, unsigned long long target) {
    const int tid = threadIdx.x, lane = tid % kWarp;
    if (tid == 0)
        sh.owner = -1;
    __syncthreads();
    if (sh.stripe_prefix[tid] <= target && target < sh.stripe_prefix[tid] + sh.stripe_weight[tid])
        sh.owner = tid;
    __syncthreads();
    const int owner = sh.owner;
    if (owner < 0)
        return -1;
    if (tid < kStripe) {
        const int v = owner * kStripe + tid;
        sh.tile[tid] = v < vocab ? weight(probs[v], v) : 0ull;
    }
    __syncthreads();
    if (tid < kWarp) {  // one warp: each lane sums eight consecutive entries, the lanes scan, the owner lane walks
        unsigned long long mine = 0;
        for (int i = 0; i < kStripe / kWarp; ++i)
            mine += sh.tile[lane * (kStripe / kWarp) + i];
        unsigned long long inclusive = mine;
        for (int offset = 1; offset < kWarp; offset <<= 1) {
            const unsigned long long other = __shfl_up_sync(0xffffffffu, inclusive, offset);
            if (lane >= offset)
                inclusive += other;
        }
        const unsigned long long before = sh.stripe_prefix[owner] + inclusive - mine;
        if (before <= target && target < before + mine) {
            unsigned long long running = before;
            for (int i = 0; i < kStripe / kWarp; ++i) {
                running += sh.tile[lane * (kStripe / kWarp) + i];
                if (running > target) {
                    sh.found = owner * kStripe + lane * (kStripe / kWarp) + i;
                    break;
                }
            }
        }
    }
    __syncthreads();
    return sh.found;
}

// Radix select over the bits of p (positive floats order like their bits), six bits per pass and the last two in the
// sixth. By count: the top `target` tokens. By mass: the shortest prefix whose fixed-point mass reaches `target`. Ties
// at the boundary value are resolved by index.
template <bool kByMass>
__device__ Boundary select_boundary(Shared& sh, const float* probs, int vocab, unsigned long long target) {
    const int tid = threadIdx.x, lane = tid % kWarp;
    constexpr int kShift[6] = {26, 20, 14, 8, 2, 0};
    constexpr int kWidth[6] = {6, 6, 6, 6, 6, 2};
    if (tid == 0) {
        sh.prefix = 0;
        sh.remaining = target;
    }
    __syncthreads();
    bool reached = true;
    for (int pass = 0; pass < 6; ++pass) {
        const int shift = kShift[pass], width = kWidth[pass];
        for (int i = tid; i < kLanes * 64; i += kThreads) {
            sh.count[i / 64][i % 64] = 0;
            sh.mass[i / 64][i % 64] = 0;
        }
        __syncthreads();
        const uint32_t prefix = sh.prefix;
        // four 16-byte loads are in flight per thread before any histogram update waits on them
        for (int base = tid; base < vocab / 4; base += 4 * kThreads) {
            float4 batch[4];
#pragma unroll
            for (int k = 0; k < 4; ++k)
                batch[k] = base + k * kThreads < vocab / 4 ? reinterpret_cast<const float4*>(probs)[base + k * kThreads]
                                                           : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if (base + k * kThreads >= vocab / 4)
                    break;
                const float elements[4] = {batch[k].x, batch[k].y, batch[k].z, batch[k].w};
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float p = elements[e];
                    const uint32_t key = __float_as_uint(p);
                    if (pass == 0 || (key >> (shift + width)) == prefix) {
                        const uint32_t bin = (key >> shift) & ((1u << width) - 1u);
                        atomicAdd(&sh.count[lane][bin], 1u);
                        if (kByMass)
                            atomicAdd(&sh.mass[lane][bin], fixed_mass(p));
                    }
                }
            }
        }
        __syncthreads();
        if (tid < 64) {
            unsigned int c = 0;
            unsigned long long m = 0;
            for (int l = 0; l < kLanes; ++l)
                c += sh.count[l][tid], m += sh.mass[l][tid];
            sh.count_total[tid] = c;
            sh.mass_total[tid] = m;
        }
        __syncthreads();
        if (tid == 0) {
            sh.chosen = 0xffffffffu;
            for (int bin = (1 << width) - 1; bin >= 0; --bin) {
                const unsigned long long amount = kByMass ? sh.mass_total[bin] : sh.count_total[bin];
                if (amount >= sh.remaining && sh.count_total[bin] > 0) {
                    sh.chosen = static_cast<unsigned>(bin);
                    break;
                }
                sh.remaining -= amount;
            }
            if (sh.chosen != 0xffffffffu) {
                sh.prefix = (pass == 0 ? 0u : sh.prefix << width) | sh.chosen;
                if (pass == 5)
                    sh.ties = sh.count_total[sh.chosen];
            }
        }
        __syncthreads();
        if (sh.chosen == 0xffffffffu) {
            reached = false;  // the target exceeds everything: keep all tokens
            break;
        }
    }
    if (!reached)
        return {0, 0, false};
    // `remaining` is what the tied tokens (p == the boundary value) must still supply: a count, or a mass.
    if (tid == 0) {
        unsigned long long need = sh.remaining;
        if (kByMass) {
            const unsigned long long each = fixed_mass(__uint_as_float(sh.prefix));
            need = each == 0 ? sh.ties : (sh.remaining + each - 1) / each;
        }
        sh.remaining_count = static_cast<unsigned>(need < 1 ? 1 : (need > sh.ties ? sh.ties : need));
    }
    __syncthreads();
    const uint32_t tied = sh.prefix;
    const auto is_tied = [tied](float p, int) { return static_cast<unsigned long long>(__float_as_uint(p) == tied); };
    scan_stripes(sh, probs, vocab, is_tied);
    const int last = locate(sh, probs, vocab, is_tied, sh.remaining_count - 1ull);
    __syncthreads();  // sh is reused by the next selection
    return {tied, last, true};
}

// One block per column.
__global__ void __launch_bounds__(kThreads)
    sample_column_kernel(const float* logits, int vocab, const int32_t* counts, const int32_t* drafts,
                         SamplingParams params, const int32_t* position, float* probs_all, int32_t* sampled) {
    __shared__ Shared sh;
    const int column = blockIdx.x, tid = threadIdx.x;
    const float* x_in = logits + int64_t{column} * vocab;
    float* probs = probs_all + int64_t{column} * vocab;
    const bool greedy = params.temperature == 0.0f;

    // 1-2. penalties and temperature, into probs; the largest value on the way.
    float peak = -INFINITY, best_value = -INFINITY;
    int best_index = 0x7fffffff;
    for (int v4 = tid; v4 < vocab / 4; v4 += kThreads) {
        const float4 four = reinterpret_cast<const float4*>(x_in)[v4];
        const int4 history = counts != nullptr ? reinterpret_cast<const int4*>(counts)[v4] : make_int4(0, 0, 0, 0);
        float elements[4] = {four.x, four.y, four.z, four.w};
        const int base_counts[4] = {history.x, history.y, history.z, history.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int v = v4 * 4 + e;
            float x = elements[e];
            int n = base_counts[e];
            for (int j = 0; j < column; ++j)
                n += drafts[j] == v;
            if (n > 0) {
                if (params.repetition_penalty != 1.0f)
                    x = x > 0.0f ? x / params.repetition_penalty : x * params.repetition_penalty;
                x = x - params.frequency_penalty * static_cast<float>(n);
                x = x - params.presence_penalty;
            }
            if (!greedy)
                x = x / params.temperature;
            elements[e] = x;
            if (x > best_value) {  // ascending v, so the first (lowest) index of the thread's largest wins
                best_value = x;
                best_index = v;
            }
            peak = fmaxf(peak, x);
        }
        reinterpret_cast<float4*>(probs)[v4] = make_float4(elements[0], elements[1], elements[2], elements[3]);
    }

    if (greedy) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_value = __shfl_xor_sync(0xffffffffu, best_value, offset);
            const int other_index = __shfl_xor_sync(0xffffffffu, best_index, offset);
            if (other_value > best_value || (other_value == best_value && other_index < best_index)) {
                best_value = other_value;
                best_index = other_index;
            }
        }
        if (tid % kWarp == 0) {
            sh.best_value[tid / kWarp] = best_value;
            sh.best_index[tid / kWarp] = best_index;
        }
        __syncthreads();
        if (tid == 0) {
            float value = sh.best_value[0];
            int index = sh.best_index[0];
            for (int w = 1; w < kThreads / kWarp; ++w)
                if (sh.best_value[w] > value || (sh.best_value[w] == value && sh.best_index[w] < index)) {
                    value = sh.best_value[w];
                    index = sh.best_index[w];
                }
            sampled[column] = index == 0x7fffffff ? 0 : index;
        }
        return;
    }

    // 3. softmax.
    peak = block_max(peak);
    float sum = 0.0f;
    for (int v4 = tid; v4 < vocab / 4; v4 += kThreads) {
        const float4 x = reinterpret_cast<const float4*>(probs)[v4];
        const float4 e = make_float4(expf(x.x - peak), expf(x.y - peak), expf(x.z - peak), expf(x.w - peak));
        reinterpret_cast<float4*>(probs)[v4] = e;
        sum += e.x, sum += e.y, sum += e.z, sum += e.w;
    }
    sum = block_sum(sum);
    for (int v4 = tid; v4 < vocab / 4; v4 += kThreads) {
        const float4 e = reinterpret_cast<const float4*>(probs)[v4];
        reinterpret_cast<float4*>(probs)[v4] = make_float4(e.x / sum, e.y / sum, e.z / sum, e.w / sum);
    }
    __syncthreads();
    const float p_max = 1.0f / sum;

    // 4. top-k and top-p boundaries.
    Boundary by_count = {0, 0, false}, by_mass = {0, 0, false};
    if (params.top_k > 0 && params.top_k < vocab)
        by_count = select_boundary<false>(sh, probs, vocab, static_cast<unsigned long long>(params.top_k));
    if (params.top_p < 1.0f) {
        unsigned long long mine = 0;
        for (int v4 = tid; v4 < vocab / 4; v4 += kThreads) {
            const float4 p = reinterpret_cast<const float4*>(probs)[v4];
            mine += fixed_mass(p.x) + fixed_mass(p.y) + fixed_mass(p.z) + fixed_mass(p.w);
        }
        const unsigned long long total = block_sum_u64(sh, mine);
        const unsigned long long target =
            max(1ull, static_cast<unsigned long long>(static_cast<double>(params.top_p) * static_cast<double>(total)));
        by_mass = select_boundary<true>(sh, probs, vocab, target);
    }
    const float min_p_threshold = params.min_p > 0.0f ? params.min_p * p_max : 0.0f;

    // 5-6. the draw: inverse transform over the kept tokens in increasing index order, in fixed-point mass.
    const auto kept_mass = [&](float p, int v) -> unsigned long long {
        const bool kept = by_count.keeps(p, v) && by_mass.keeps(p, v) && p >= min_p_threshold && p > 0.0f;
        return kept ? fixed_mass(p) : 0ull;
    };
    const unsigned long long total = scan_stripes(sh, probs, vocab, kept_mass);
    const double u =
        static_cast<double>(uniform(params.seed, static_cast<uint32_t>(*position + column), kPurposeSample));
    unsigned long long target = static_cast<unsigned long long>(u * static_cast<double>(total));
    if (target >= total)
        target = total - 1;
    const int token = locate(sh, probs, vocab, kept_mass, target);
    if (tid == 0)
        sampled[column] = token >= 0 ? token : 0;
}

__global__ void accept_kernel(const int32_t* sampled, const int32_t* drafts, int columns, int32_t* accepted,
                              int32_t* tokens) {
    int k = 0;
    while (k < columns - 1 && sampled[k] == drafts[k])
        ++k;
    for (int i = 0; i < columns; ++i)
        tokens[i] = i <= k ? sampled[i] : -1;
    *accepted = k + 1;
}

}  // namespace

int64_t sampling_scratch_floats(int vocab, int columns) {
    return int64_t{vocab} * columns;
}

void sample_columns(const float* logits, int vocab, int columns, const int32_t* counts, const int32_t* drafts,
                    const SamplingParams& params, const int32_t* position, float* scratch, int32_t* sampled,
                    cudaStream_t stream) {
    if (vocab % 4 != 0 || vocab > kStripe * kStripes)
        throw Error("sample_columns needs a vocabulary that is a multiple of 4 and at most 262,144.");
    sample_column_kernel<<<columns, kThreads, 0, stream>>>(logits, vocab, counts, drafts, params, position, scratch,
                                                           sampled);
}

void accept_drafts(const int32_t* sampled, const int32_t* drafts, int columns, int32_t* accepted, int32_t* tokens,
                   cudaStream_t stream) {
    accept_kernel<<<1, 1, 0, stream>>>(sampled, drafts, columns, accepted, tokens);
}

}  // namespace torch::kernels
