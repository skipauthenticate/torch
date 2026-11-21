set(fixtures ${CMAKE_CURRENT_SOURCE_DIR}/fixtures)

add_executable(dequant_test unit/dequant_test.cpp)
target_link_libraries(dequant_test PRIVATE torch_core)
add_test(NAME dequant COMMAND dequant_test ${fixtures}/quant_blocks.bin)

add_executable(gguf_test unit/gguf_test.cpp)
target_link_libraries(gguf_test PRIVATE torch_core)
add_test(NAME gguf COMMAND gguf_test ${fixtures}/tiny.gguf)

foreach(name budget ple_rows window_policy swaps tokenizer_export prefix_index)
  add_executable(${name}_test unit/${name}_test.cpp)
  target_link_libraries(${name}_test PRIVATE torch_core)
  add_test(NAME ${name} COMMAND ${name}_test)
endforeach()

add_executable(placement_test unit/placement_test.cpp)
target_link_libraries(placement_test PRIVATE torch_core)
add_test(NAME placement COMMAND placement_test ${CMAKE_CURRENT_BINARY_DIR})

add_executable(direct_io_test unit/direct_io_test.cpp)
target_link_libraries(direct_io_test PRIVATE torch_core)
add_test(NAME direct_io COMMAND direct_io_test ${CMAKE_CURRENT_BINARY_DIR})

# Give the test control over the completion of one real disk read.
add_library(hedged_test_io OBJECT ../src/runtime/direct_io.cpp)
target_link_libraries(hedged_test_io PRIVATE torch_core)
target_compile_definitions(hedged_test_io PRIVATE pread=torch_test_pread)
# Glibc's fortified pread (on by default in Ubuntu's compiler) would call the real pread and skip the test's.
target_compile_options(hedged_test_io PRIVATE -U_FORTIFY_SOURCE)
add_executable(hedged_reads_test unit/hedged_reads_test.cpp $<TARGET_OBJECTS:hedged_test_io>)
target_link_libraries(hedged_reads_test PRIVATE torch_core)
add_test(NAME hedged_reads COMMAND hedged_reads_test ${CMAKE_CURRENT_BINARY_DIR})

add_test(NAME inspect_json COMMAND torch inspect --json --tensors ${fixtures}/tiny.gguf)
set_tests_properties(inspect_json PROPERTIES PASS_REGULAR_EXPRESSION "\"general.architecture\": \"qwen4exp\"")

# The first 600 bytes of tiny.gguf stop inside its tensor data: readable with --header-only, rejected without.
set(cut ${CMAKE_CURRENT_BINARY_DIR}/tiny-600.gguf)
add_test(NAME inspect_header_only
         COMMAND sh -c "head -c 600 \"$0\" > \"$1\" && \"$2\" inspect --header-only \"$1\" && ! \"$2\" inspect \"$1\""
                 ${fixtures}/tiny.gguf ${cut} $<TARGET_FILE:torch>)

set_tests_properties(prefix_index dequant gguf budget placement ple_rows window_policy swaps tokenizer_export direct_io hedged_reads inspect_json inspect_header_only PROPERTIES LABELS unit)

if(TORCH_FUZZ)
  add_executable(gguf_fuzz fuzz/gguf_fuzz.cpp)
  target_link_libraries(gguf_fuzz PRIVATE torch_core)
  target_link_options(gguf_fuzz PRIVATE -fsanitize=fuzzer)
endif()

# The C API without a model, for torch_server's tests on any machine (tests/server/test_server.py).
enable_language(C)
add_library(torch_engine_stub SHARED server/stub_engine.c)
set_target_properties(torch_engine_stub PROPERTIES C_STANDARD 11 C_VISIBILITY_PRESET hidden)
target_include_directories(torch_engine_stub PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_options(torch_engine_stub PRIVATE -Wall -Wextra $<$<BOOL:${TORCH_WERROR}>:-Werror>)

# GPU kernels against their CPU references (synthetic data; the Orin only).
if(TORCH_CUDA)
  foreach(name gemv gemm norm hc deltanet attention moe ple argmax gate sample indexer)
    add_executable(${name}_test parity/${name}_test.cpp)
    target_link_libraries(${name}_test PRIVATE torch_engine)
    add_test(NAME ${name} COMMAND ${name}_test)
    set_tests_properties(${name} PROPERTIES LABELS parity)
  endforeach()
  # The C API on the real model, run by hand on the Orin (tests/api/torch_engine_test.c): a C program against
  # include/torch_engine.h and libtorch_engine.so.
  enable_language(C)
  find_package(Threads REQUIRED)
  add_executable(prefix_disk_test parity/prefix_disk_test.cpp)
  target_link_libraries(prefix_disk_test PRIVATE torch_engine)
  add_test(NAME prefix_disk COMMAND prefix_disk_test)
  set_tests_properties(prefix_disk PROPERTIES LABELS parity)

  add_executable(copy_window_test model/copy_window_test.cpp)
  target_link_libraries(copy_window_test PRIVATE torch_engine)

  add_executable(torch_engine_test api/torch_engine_test.c)
  set_target_properties(torch_engine_test PROPERTIES C_STANDARD 11)
  target_compile_options(torch_engine_test PRIVATE -Wall -Wextra $<$<BOOL:${TORCH_WERROR}>:-Werror>)
  target_link_libraries(torch_engine_test PRIVATE torch_engine_shared Threads::Threads)
endif()
