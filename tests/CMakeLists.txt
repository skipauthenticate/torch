set(fixtures ${CMAKE_CURRENT_SOURCE_DIR}/fixtures)

add_executable(dequant_test unit/dequant_test.cpp)
target_link_libraries(dequant_test PRIVATE torch_core)
add_test(NAME dequant COMMAND dequant_test ${fixtures}/quant_blocks.bin)

add_executable(gguf_test unit/gguf_test.cpp)
target_link_libraries(gguf_test PRIVATE torch_core)
add_test(NAME gguf COMMAND gguf_test ${fixtures}/tiny.gguf)

add_test(NAME inspect_json COMMAND torch inspect --json --tensors ${fixtures}/tiny.gguf)
set_tests_properties(inspect_json PROPERTIES PASS_REGULAR_EXPRESSION "\"general.architecture\": \"qwen4exp\"")

# The first 600 bytes of tiny.gguf stop inside its tensor data: readable with --header-only, rejected without.
set(cut ${CMAKE_CURRENT_BINARY_DIR}/tiny-600.gguf)
add_test(NAME inspect_header_only
         COMMAND sh -c "head -c 600 \"$0\" > \"$1\" && \"$2\" inspect --header-only \"$1\" && ! \"$2\" inspect \"$1\""
                 ${fixtures}/tiny.gguf ${cut} $<TARGET_FILE:torch>)

set_tests_properties(dequant gguf inspect_json inspect_header_only PROPERTIES LABELS unit)

if(TORCH_FUZZ)
  add_executable(gguf_fuzz fuzz/gguf_fuzz.cpp)
  target_link_libraries(gguf_fuzz PRIVATE torch_core)
  target_link_options(gguf_fuzz PRIVATE -fsanitize=fuzzer)
endif()
