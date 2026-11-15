// gemm_bf16_tc (gemm.h): prefill's BF16 products on the tensor cores, Options::fast_prefill (torch run
// --fast-prefill, torch_server --fast-prefill, TORCH_FAST_PREFILL=1). The activations are rounded to BF16 (round to
// nearest even) as a block stages them, and each 16 x 8 output tile accumulates in F32 over K in order, 16 at a time
// (mma.sync m16n8k16). Each output's sum depends only on its row's weights, its column's activations and K, never on
// how many columns the launch has: deterministic, and the same for any chunk size. Not bitwise gemm_bf16.
#include <cuda_bf16.h>

#include <cstdint>

#include "kernels/common.cuh"
#include "kernels/gemm/gemm.h"

namespace torch::kernels {
namespace {

// 128 x 64 tiles, K 64 at a time: the BF16 stage of a 4,096-token prompt 879 ms, against 1,042 at 64 x 64 x 32, 963 at
// 64 x 64 x 64, 1,268 at 128 x 64 x 32 and 1,307 at 256 x 64 x 64 (the same bits in all: K runs in order either way).
constexpr int kTcM = 128, kTcN = 64, kTcK = 64, kTcPad = 8, kTcThreads = 256;

__device__ __forceinline__ uint32_t pack_bf16(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);  // a in the low half
    return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* p) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Rows [0, rows) from w into y (with SiLU of scale x the sum when scale != 0), rows [rows, rows + rows2) from w2 into
// y2; y[c * rows + r] for column c, as gemm_bf16. A block takes kTcM rows x 64 columns, its warps 32 x 32 each.
__global__ void __launch_bounds__(kTcThreads)
    gemm_bf16_tc_kernel(const uint8_t* w, int rows, const uint8_t* w2, int rows2, int K, const float* x, int N,
                        float* y, float* y2, float scale) {
    __shared__ __align__(16) uint16_t as[kTcM][kTcK + kTcPad];
    __shared__ __align__(16) uint16_t bs[kTcN][kTcK + kTcPad];
    const int M = rows + rows2, m0 = static_cast<int>(blockIdx.x) * kTcM, n0 = static_cast<int>(blockIdx.y) * kTcN;
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const int wm = warp / 2 * 32, wn = warp % 2 * 32;
    float acc[2][4][4] = {};
    for (int k0 = 0; k0 < K; k0 += kTcK) {
        for (int i = static_cast<int>(threadIdx.x); i < kTcM * kTcK / 8; i += kTcThreads) {
            const int r = i / (kTcK / 8), q = i % (kTcK / 8), gr = m0 + r;
            uint4 v{0, 0, 0, 0};
            if (gr < M) {
                const uint8_t* row =
                    gr < rows ? w + static_cast<int64_t>(gr) * K * 2 : w2 + static_cast<int64_t>(gr - rows) * K * 2;
                v = *reinterpret_cast<const uint4*>(row + static_cast<int64_t>(k0 + 8 * q) * 2);
            }
            *reinterpret_cast<uint4*>(&as[r][8 * q]) = v;
        }
        for (int i = static_cast<int>(threadIdx.x); i < kTcN * kTcK / 8; i += kTcThreads) {
            const int c = i / (kTcK / 8), q = i % (kTcK / 8), gc = n0 + c;
            float4 a{0, 0, 0, 0}, b{0, 0, 0, 0};
            if (gc < N) {
                const auto* p = reinterpret_cast<const float4*>(x + static_cast<int64_t>(gc) * K + k0 + 8 * q);
                a = p[0], b = p[1];
            }
            *reinterpret_cast<uint4*>(&bs[c][8 * q]) =
                make_uint4(pack_bf16(a.x, a.y), pack_bf16(a.z, a.w), pack_bf16(b.x, b.y), pack_bf16(b.z, b.w));
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < kTcK; kk += 16) {
            uint32_t a[2][4], b[4][2];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
                ldmatrix_x4(a[mi], &as[wm + 16 * mi + lane % 16][kk + lane / 16 * 8]);
#pragma unroll
            for (int ni = 0; ni < 4; ni += 2) {
                uint32_t r[4];
                ldmatrix_x4(r, &bs[wn + 8 * ni + lane % 8 + lane / 16 * 8][kk + lane / 8 % 2 * 8]);
                b[ni][0] = r[0], b[ni][1] = r[1], b[ni + 1][0] = r[2], b[ni + 1][1] = r[3];
            }
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                for (int ni = 0; ni < 4; ++ni)
                    mma_bf16(acc[mi][ni], a[mi], b[ni][0], b[ni][1]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int r = m0 + wm + 16 * mi + lane / 4 + 8 * h, c = n0 + wn + 8 * ni + lane % 4 * 2 + j;
                    if (c >= N || r >= M)
                        continue;
                    const float v = acc[mi][ni][2 * h + j];
                    if (r < rows)
                        y[static_cast<int64_t>(c) * rows + r] = scale != 0.0f ? silu(v * scale) : v;
                    else
                        y2[static_cast<int64_t>(c) * rows2 + r - rows] = v;
                }
}

}  // namespace

bool gemm_bf16_tc(const uint8_t* w, int rows, int cols, const float* x, float* y, int columns, float silu_scale,
                  cudaStream_t stream, const uint8_t* w2, int rows2, float* y2) {
    if (cols % kTcK != 0 || rows < 1 || columns < 1 || rows2 < 0)
        return false;
    const dim3 grid(static_cast<unsigned>((rows + rows2 + kTcM - 1) / kTcM),
                    static_cast<unsigned>((columns + kTcN - 1) / kTcN));
    gemm_bf16_tc_kernel<<<grid, kTcThreads, 0, stream>>>(w, rows, w2, rows2, cols, x, columns, y, y2, silu_scale);
    return true;
}

}  // namespace torch::kernels
