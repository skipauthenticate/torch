// The per-layer embedding (PLE) block for one token. The residual is hc streams of n floats (res[s * n + e]); a
// per-stream gate scales a shared value vector, and a dilated causal convolution over the normalized values (taps at
// 9, 6, 3 and 0 positions back) is added to every stream.
//
// Parity test: tests/parity/ple_test.cpp (ple_conv over 12 consecutive positions, so the 9-row ring wraps).
// Bytes moved: ple_conv reads gated, x and res and writes res, reads and writes one history row, and reads the
// weights: about 8 floats per channel.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace torch::kernels {

// Per stream s: v = (sum_i key_n[s * n + i] * query[s * n + i]) * (1.0f / sqrtf(n));
// gate[s] = sigmoid(sign(v) * sqrtf(fmaxf(fabsf(v), 1e-6f))), with sign(0) = 0. The sum runs in a fixed block order.
void ple_gate(const float* key_n, const float* query, float* gate, int n, int hc, cudaStream_t stream);

// gated[s * n + e] = value[e] * gate[s].
void ple_gated_value(const float* value, const float* gate, float* gated, int n, int hc, cudaStream_t stream);

// channels = hc * n. hist is a ring of 9 rows [9][channels]; the row of position p is p % 9. With back(b) = 0 when
// *pos - b < 0 and hist[((*pos - b) % 9) * channels + c] otherwise:
//   conv = w0 * back(9) + w1 * back(6) + w2 * back(3) + w3 * x[c]   (w is [channels][4]; summed in that order)
// then hist row *pos % 9 = x[c] and res[c] += gated[c] + silu(conv).
// Tap 9 reads the row that this call is about to overwrite, which is why the read comes first.
void ple_conv(float* res, const float* gated, const float* x, float* hist, const float* w, int channels,
              const int32_t* pos, cudaStream_t stream);

// A window of `columns` (1 to 8) tokens at positions *pos .. *pos + columns - 1: gated, x and res are
// [columns][channels]. Column t does what ple_conv does at position *pos + t, except that a tap at position *pos + t -
// b comes from column t - b of x when that is inside the window and from the committed ring otherwise, and the ring is
// only read. A window that accepts fewer columns than it ran must not leave a rejected column's x in the ring (the next
// window's tap 9 would read it): ple_commit writes the accepted columns afterwards.
void ple_conv_window(float* res, const float* gated, const float* x, const float* hist, const float* w, int channels,
                     const int32_t* pos, int columns, cudaStream_t stream);

// hist row (*pos + t) % 9 = column t of x, for the first `accepted` (1 to 8) columns.
void ple_commit(float* hist, const float* x, int channels, const int32_t* pos, int accepted, cudaStream_t stream);

// The same with the count in device memory (an int32 holding 1 to 8), so the commit can be captured in a graph.
void ple_commit(float* hist, const float* x, int channels, const int32_t* pos, const int32_t* accepted,
                cudaStream_t stream);

}  // namespace torch::kernels
