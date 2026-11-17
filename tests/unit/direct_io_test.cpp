// Checks direct reads at unaligned offsets, at the end of a file and out of range, reads of a model split into shard
// files (gguf_writer.h) at its tensors' offsets, that a shard failing to open leaves no file open, and the read pool.
//
// usage: direct_io_test DIR (a directory on a disk file system: tmpfs does not support O_DIRECT)
#include "runtime/direct_io.h"

#include <fcntl.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "check.h"
#include "core/error.h"
#include "gguf/gguf.h"
#include "gguf_writer.h"

using namespace torch;
using runtime::DirectFile;
using runtime::kPage;
using runtime::span_bytes;

namespace {

void check_reads(const std::filesystem::path& dir) {
    // Three pages and 100 bytes, byte i = i * 7 mod 251, so every offset has a recognizable value.
    std::vector<uint8_t> bytes(3 * kPage + 100);
    for (size_t i = 0; i < bytes.size(); ++i)
        bytes[i] = static_cast<uint8_t>(i * 7 % 251);
    const std::filesystem::path path = dir / "direct_io_test.bin";
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), std::ssize(bytes));

    const DirectFile f(path);
    CHECK(f.size() == bytes.size());
    auto* buffer = static_cast<uint8_t*>(std::aligned_alloc(kPage, 4 * kPage));
    struct Case {
        uint64_t offset, n;
    };
    for (Case c : {Case{0, 10}, Case{4095, 2}, Case{100, 3 * kPage}, Case{3 * kPage + 50, 50}, Case{0, 3 * kPage + 100},
                   Case{kPage, 0}}) {
        CHECK(span_bytes(c.offset, c.n) <= 4 * kPage);
        const uint8_t* got = f.read(c.offset, c.n, buffer);
        CHECK(got == buffer + c.offset % kPage);
        CHECK(std::memcmp(got, bytes.data() + c.offset, c.n) == 0);
    }
    bool threw = false;
    try {
        f.read(3 * kPage + 50, 51, buffer);  // one byte past the end
    } catch (const Error&) {
        threw = true;
    }
    CHECK(threw);
    std::free(buffer);
    std::filesystem::remove(path);
}

// A split model's tensors read at data_offset() + offset, whichever shard holds them, in both layouts; b ends its
// shard's file inside a page; a range that leaves a file throws.
void check_model_reads(const std::filesystem::path& dir, const test::SplitLayout& layout) {
    const std::filesystem::path first = test::split_model(dir, "direct_io_model", layout);
    const gguf::File m = gguf::File::open_model(first);
    const DirectFile f(m);
    CHECK(f.size() == m.file_size());
    auto* buffer = static_cast<uint8_t*>(std::aligned_alloc(kPage, 4 * kPage));
    for (const test::SplitTensor& want : test::split_tensors(layout)) {
        const gguf::TensorInfo* t = m.tensor(want.name);
        CHECK(t != nullptr);
        if (t == nullptr)
            continue;
        const auto* bytes = reinterpret_cast<const uint8_t*>(want.values.data());
        struct Case {
            uint64_t from, n;
        };
        for (Case c : {Case{0, t->nbytes}, Case{4, 8}, Case{t->nbytes - 4, 4}}) {
            CHECK(span_bytes(m.data_offset() + t->offset + c.from, c.n) <= 4 * kPage);
            const uint8_t* got = f.read(m.data_offset() + t->offset + c.from, c.n, buffer);
            CHECK(std::memcmp(got, bytes + c.from, c.n) == 0);
        }
    }
    const auto throws = [&](uint64_t offset, uint64_t n) {
        try {
            f.read(offset, n, buffer);
            return false;
        } catch (const Error&) {
            return true;
        }
    };
    const gguf::Part &before = m.parts()[m.parts().size() - 2], &last = m.parts().back();
    CHECK(throws(before.base + before.size - 4, 8));      // across the end of a shard
    CHECK(throws(before.base + before.size, 1));          // between two shards
    CHECK(!throws(last.base, 4) && throws(f.size(), 1));  // the start of the last shard; past it
    std::free(buffer);
    for (const gguf::Part& p : m.parts())
        std::filesystem::remove(p.path);
}

void check_pool() {
    runtime::ReadPool pool(4);
    std::vector<std::atomic<int>> seen(1000);
    for (int round = 0; round < 3; ++round) {
        pool.run(1000, [&](int i) { seen[static_cast<size_t>(i)]++; });
        pool.run(0, [](int) {});
    }
    bool all = true;
    for (const auto& s : seen)
        all = all && s.load() == 3;
    CHECK(all);
    bool threw = false;
    try {
        pool.run(10, [](int i) {
            if (i == 7)
                throw Error("job 7");
        });
    } catch (const Error& e) {
        threw = std::strcmp(e.what(), "job 7") == 0;
    }
    CHECK(threw);
    pool.run(5, [&](int i) { seen[static_cast<size_t>(i)]++; });  // still usable after an exception
    CHECK(seen[4].load() == 4);
}

}  // namespace

int open_fds() {
    int n = 0;
    for (int fd = 0; fd < 1024; ++fd)
        n += fcntl(fd, F_GETFD) != -1;
    return n;
}

// The last shard cannot be opened after the others were: the constructor throws and closes the files it opened.
void check_open_failure(const std::filesystem::path& dir) {
    const gguf::File m = gguf::File::open_model(test::split_model(dir, "direct_io_missing"));
    std::filesystem::remove(m.parts().back().path);
    const int before = open_fds();
    bool threw = false;
    try {
        const DirectFile f(m);
    } catch (const Error&) {
        threw = true;
    }
    CHECK(threw);
    CHECK(open_fds() == before);
    for (const gguf::Part& p : m.parts())
        std::filesystem::remove(p.path);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: direct_io_test DIR\n");
        return 2;
    }
    check_reads(argv[1]);
    check_model_reads(argv[1], {});
    check_model_reads(argv[1], {.weights_in_shard1 = true});
    check_open_failure(argv[1]);
    check_pool();
    return test::test_exit_code();
}
