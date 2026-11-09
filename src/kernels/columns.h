// The column count of a verify window, shared by host and device code: kernels take 1 to kMaxColumns columns, each
// bitwise the one-column result (gemv.h).
#pragma once

namespace torch::kernels {

inline constexpr int kMaxColumns = 8;

}  // namespace torch::kernels
