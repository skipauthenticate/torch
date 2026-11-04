// libFuzzer target: any input either parses or throws torch::Error. A crash, a sanitizer report or any other exception
// is a bug in the reader.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/error.h"
#include "gguf/gguf.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        const auto f = torch::gguf::File::parse(std::vector<uint8_t>(data, data + size));
        // Decode every value and size every tensor, so the fuzzer also reaches the accessors.
        for (const auto& [key, value] : f.metadata()) {
            (void)f.get<uint64_t>(key);
            (void)f.get<double>(key);
            (void)f.get<std::string>(key);
            (void)f.get<std::vector<std::string>>(key);
            (void)f.get<std::vector<uint64_t>>(key);
        }
        for (const auto& t : f.tensors()) {
            (void)t.elements();
            (void)t.row_bytes();
            (void)t.slice_offset(t.shape[t.n_dims - 1] - 1);
        }
    } catch (const torch::Error&) {
    }
    return 0;
}
