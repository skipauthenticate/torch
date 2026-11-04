// Scalar activation functions in double, for the CPU references of the kernel parity tests.
#pragma once

#include <cmath>

namespace torch::test {

inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}
inline double silu(double x) {
    return x * sigmoid(x);
}
inline double softplus(double x) {
    return std::log1p(std::exp(x));
}

}  // namespace torch::test
