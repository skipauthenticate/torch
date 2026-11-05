set(fixtures ${CMAKE_CURRENT_SOURCE_DIR}/fixtures)

add_executable(dequant_test unit/dequant_test.cpp)
target_link_libraries(dequant_test PRIVATE torch_core)
add_test(NAME dequant COMMAND dequant_test ${fixtures}/quant_blocks.bin)

add_executable(gguf_test unit/gguf_test.cpp)
target_link_libraries(gguf_test PRIVATE torch_core)
add_test(NAME gguf COMMAND gguf_test ${fixtures}/tiny.gguf)

foreach(name budget ple_rows)
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

add_executable(hedged_reads_test unit/hedged_reads_test.cpp)
target_link_libraries(hedged_reads_test PRIVATE torch_core)
add_test(NAME hedged_reads COMMAND hedged_reads_test ${CMAKE_CURRENT_BINARY_DIR})

add_test(NAME inspect_json COMMAND torch inspect --json --tensors ${fixtures}/tiny.gguf)
set_tests_properties(inspect_json PROPERTIES PASS_REGULAR_EXPRESSION "\"general.architecture\": \"qwen4exp\"")

# The first 600 bytes of tiny.gguf stop inside its tensor data: readable with --header-only, rejected without.
set(cut ${CMAKE_CURRENT_BINARY_DIR}/tiny-600.gguf)
add_test(NAME inspect_header_only
         COMMAND sh -c "head -c 600 \"$0\" > \"$1\" && \"$2\" inspect --header-only \"$1\" && ! \"$2\" inspect \"$1\""
                 ${fixtures}/tiny.gguf ${cut} $<TARGET_FILE:torch>)

set_tests_properties(dequant gguf budget placement ple_rows direct_io hedged_reads inspect_json inspect_header_only PROPERTIES LABELS unit)

if(TORCH_FUZZ)
  add_executable(gguf_fuzz fuzz/gguf_fuzz.cpp)
  target_link_libraries(gguf_fuzz PRIVATE torch_core)
  target_link_options(gguf_fuzz PRIVATE -fsanitize=fuzzer)
endif()

# GPU kernels against their CPU references (synthetic data; the Orin only).
if(TORCH_CUDA)
  foreach(name gemv norm hc deltanet attention moe ple argmax)
    add_executable(${name}_test parity/${name}_test.cpp)
    target_link_libraries(${name}_test PRIVATE torch_engine)
    add_test(NAME ${name} COMMAND ${name}_test)
    set_tests_properties(${name} PROPERTIES LABELS parity)
  endforeach()
endif()
