// Checks direct reads at unaligned offsets, at the end of a file and out of range, and the read pool.
//
// usage: direct_io_test DIR (a directory on a disk file system: tmpfs does not support O_DIRECT)
#include "runtime/direct_io.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "check.h"
#include "core/error.h"

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

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: direct_io_test DIR\n");
        return 2;
    }
    check_reads(argv[1]);
    check_pool();
    return test::test_exit_code();
}
