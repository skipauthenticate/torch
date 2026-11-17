// libFuzzer target: any input either parses or throws torch::Error, alone and as shard 2 of a split model whose shard
// 1 is fixed (File::open_model). A crash, a sanitizer report or any other exception is a bug in the reader.
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../unit/gguf_writer.h"
#include "core/error.h"
#include "gguf/gguf.h"

namespace {

// A folder with f-00001-of-00002.gguf: metadata only, saying the two shards hold 2 tensors. One per process, so fuzzing
// jobs that run side by side do not write each other's shard 2.
const std::filesystem::path& split_folder() {
    static const std::filesystem::path dir = [] {
        const std::filesystem::path d =
            std::filesystem::temp_directory_path() / ("torch-gguf-fuzz-split-" + std::to_string(getpid()));
        std::filesystem::create_directories(d);
        torch::test::Writer().header(0, 3).split(0, 2, 2).save(d / "f-00001-of-00002.gguf");
        return d;
    }();
    return dir;
}

}  // namespace

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
    try {
        std::ofstream(split_folder() / "f-00002-of-00002.gguf", std::ios::binary)
            .write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
        const auto m = torch::gguf::File::open_model(split_folder() / "f-00001-of-00002.gguf");
        for (const auto& t : m.tensors())
            (void)(t.offset + t.nbytes);
    } catch (const torch::Error&) {
    }
    return 0;
}
