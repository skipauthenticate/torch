cmake_minimum_required(VERSION 3.24)
project(torch VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  set(torch_cuda_default ON)
else()
  set(torch_cuda_default OFF)
endif()
option(TORCH_CUDA "Build the GPU engine (ON when a CUDA compiler is found)" ${torch_cuda_default})
option(TORCH_TESTS "Build the tests" ON)
option(TORCH_FUZZ "Build the fuzz targets (Clang with libFuzzer)" OFF)
option(TORCH_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(TORCH_WERROR "Treat warnings as errors" OFF)

# A tree copied without .git can pass the commit with -DTORCH_GIT_SHA=<sha>.
if(NOT TORCH_GIT_SHA)
  execute_process(COMMAND git rev-parse --short=12 HEAD
                  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                  OUTPUT_VARIABLE TORCH_GIT_SHA
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
if(NOT TORCH_GIT_SHA)
  set(TORCH_GIT_SHA unknown)
endif()
configure_file(cmake/version.h.in ${PROJECT_BINARY_DIR}/generated/torch/version.h @ONLY)

add_library(torch_warnings INTERFACE)
target_compile_options(torch_warnings INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wshadow;-Wconversion;-Wno-sign-conversion>
  $<$<AND:$<BOOL:${TORCH_WERROR}>,$<COMPILE_LANGUAGE:CXX>>:-Werror>)

if(TORCH_SANITIZE)
  # A sanitizer report ends the process, so a test that triggers one fails.
  add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

if(TORCH_FUZZ)
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "TORCH_FUZZ needs LLVM Clang, which ships libFuzzer; Apple Clang and GCC do not.")
  endif()
  # Coverage instrumentation for every target, so the fuzzer sees the reader's branches.
  add_compile_options(-fsanitize=fuzzer-no-link)
endif()

# Host code: no CUDA, so it builds and is unit-tested on any machine.
add_library(torch_core STATIC
  src/core/budget.cpp
  src/core/placement.cpp
  src/core/swaps.cpp
  src/formats/types.cpp
  src/gguf/gguf.cpp
  src/model/config.cpp
  src/model/window_policy.cpp
  src/runtime/direct_io.cpp
  src/runtime/hedged_reads.cpp
  src/runtime/ple_rows.cpp)
target_include_directories(torch_core PUBLIC src)
target_link_libraries(torch_core PUBLIC torch_warnings)

add_executable(torch tools/main.cpp tools/inspect.cpp)
target_include_directories(torch PRIVATE ${PROJECT_BINARY_DIR}/generated)
target_link_libraries(torch PRIVATE torch_core)

if(TORCH_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87)  # Jetson AGX Orin only
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)

  # The GPU engine: kernels (src/kernels/<operation>/), weights, the forward pass and the expert store.
  add_library(torch_engine STATIC
    src/kernels/gemv/gemv.cu
    src/kernels/attention/attention.cu
    src/kernels/deltanet/deltanet.cu
    src/kernels/hc/hc.cu
    src/kernels/moe/moe.cu
    src/kernels/norm/norm.cu
    src/kernels/ple/ple.cu
    src/kernels/sampling/argmax.cu
    src/model/draft.cpp
    src/kernels/sampling/sample.cu
    src/model/model.cpp
    src/model/weights.cpp
    src/runtime/expert_gate.cu
    src/runtime/expert_store.cpp)
  target_link_libraries(torch_engine PUBLIC torch_core CUDA::cudart)
  # nvcc warns (445) about C++20 template lambdas whose template parameter appears only in the body, which is the
  # with_type() dispatch pattern of src/formats/types.h.
  target_compile_options(torch_engine PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=445>)

  target_sources(torch PRIVATE tools/run.cpp)
  target_link_libraries(torch PRIVATE torch_engine)
  target_compile_definitions(torch PRIVATE TORCH_ENGINE)

  # Microbenchmarks: not tests, run by hand on the Orin (docs/development.md).
  add_executable(torch_bench_memory bench/memory_bandwidth.cu)
  target_link_libraries(torch_bench_memory PRIVATE torch_core CUDA::cudart)
  add_executable(torch_bench_graph_gaps bench/graph_gaps.cu)
  target_link_libraries(torch_bench_graph_gaps PRIVATE torch_core CUDA::cudart)
  add_executable(torch_bench_miss_roundtrip bench/miss_roundtrip.cu)
  target_link_libraries(torch_bench_miss_roundtrip PRIVATE torch_core CUDA::cudart)
  add_executable(torch_bench_kernels bench/kernels.cu)
  target_include_directories(torch_bench_kernels PRIVATE ${PROJECT_BINARY_DIR}/generated)
  target_link_libraries(torch_bench_kernels PRIVATE torch_engine)
  add_executable(torch_bench_ssd bench/ssd_reads.cpp)
  target_link_libraries(torch_bench_ssd PRIVATE torch_warnings)
endif()

if(TORCH_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
