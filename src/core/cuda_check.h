// Error checking for CUDA runtime calls in loading and setup code. Only CUDA targets include this header.
#pragma once

#include <cuda_runtime.h>

#include <string>

#include "core/error.h"

namespace torch {

// Throws torch::Error naming the failed call and CUDA's message.
inline void cuda_check(cudaError_t status, const char* call) {
    if (status != cudaSuccess) {
        throw Error(std::string(call) + " failed (" + cudaGetErrorString(status) +
                    "); check that the GPU is not in use by another process and that memory is free.");
    }
}

}  // namespace torch

#define TORCH_CUDA_CHECK(call) ::torch::cuda_check((call), #call)
