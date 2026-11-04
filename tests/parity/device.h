// Small helpers for the GPU parity tests: device buffers filled from and read back into std::vector, and the
// comparison every test uses.
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/cuda_check.h"

namespace torch::test {

template <class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) { TORCH_CUDA_CHECK(cudaMalloc(&data_, bytes())); }
    explicit DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size()) { upload(host); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { cudaFree(data_); }

    T* get() const { return data_; }
    size_t bytes() const { return count_ * sizeof(T); }
    void upload(const std::vector<T>& host) {
        TORCH_CUDA_CHECK(cudaMemcpy(data_, host.data(), bytes(), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> host(count_);
        TORCH_CUDA_CHECK(cudaDeviceSynchronize());
        TORCH_CUDA_CHECK(cudaMemcpy(host.data(), data_, bytes(), cudaMemcpyDeviceToHost));
        return host;
    }

private:
    T* data_ = nullptr;
    size_t count_;
};

inline std::vector<float> random_floats(size_t n, float lo, float hi, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (float& x : v)
        x = dist(rng);
    return v;
}

// Largest |got - want| / scale over the elements, where scale is the per-element magnitude the error is measured
// against (for a dot product, the sum of |products|).
inline double worst_relative(const std::vector<float>& got, const std::vector<double>& want,
                             const std::vector<double>& scale) {
    double worst = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double error = std::fabs(got[i] - want[i]) / std::fmax(scale[i], 1e-30);
        if (std::isnan(error))
            return error;  // fmax would drop a NaN and report a broken kernel as exact
        worst = std::fmax(worst, error);
    }
    return worst;
}

// Prints one result line and returns whether it passed.
inline bool report(const char* what, double error, double tolerance) {
    const bool ok = error <= tolerance;
    std::printf("%-40s max relative error %.3g (tolerance %.1g) %s\n", what, error, tolerance, ok ? "ok" : "FAIL");
    return ok;
}

}  // namespace torch::test
