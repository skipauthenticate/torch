// Checks hedged reads: every request's bytes arrive from the right offset, with and without hedging, buffers come back
// after release(), and a failed read gives nullptr. With a hedge delay of 0 almost every read is issued twice, so the
// winner-and-loser path (and late buffers) runs on every round.
//
// usage: hedged_reads_test DIR (a directory on a disk file system: tmpfs does not support O_DIRECT)
#include "runtime/hedged_reads.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

#include "check.h"

using namespace torch;
using runtime::DirectFile;
using runtime::HedgedReads;
using runtime::kPage;

namespace {

void check(const DirectFile& f, const std::vector<uint8_t>& bytes, std::chrono::microseconds hedge_after, bool hedges) {
    constexpr int kReads = 6, kBuffers = kReads + 4;
    constexpr uint64_t kBuffer = 3 * kPage;
    auto* memory = static_cast<uint8_t*>(std::aligned_alloc(kPage, 2 * kBuffers * kBuffer));
    {
        HedgedReads reads(f, 4, hedge_after);
        const int small = reads.add_pool(memory, kBuffer, kBuffers);
        const int large = reads.add_pool(memory + kBuffers * kBuffer, kBuffer, kBuffers);
        CHECK(small == 0 && large == 1);
        std::mt19937 rng(7);
        for (int round = 0; round < 200; ++round) {
            std::vector<HedgedReads::Request> requests;
            for (int i = 0; i < kReads; ++i) {
                const uint64_t n = 1 + rng() % (2 * kPage);
                requests.push_back({i % 2, rng() % (bytes.size() - n), n});
            }
            std::vector<const uint8_t*> out(kReads);
            reads.read_all(requests, out);
            for (int i = 0; i < kReads; ++i)
                CHECK(out[static_cast<size_t>(i)] != nullptr &&
                      std::memcmp(out[static_cast<size_t>(i)], bytes.data() + requests[static_cast<size_t>(i)].offset,
                                  requests[static_cast<size_t>(i)].n) == 0);
            reads.release();
        }
        CHECK(hedges ? reads.hedged() > 0 : reads.hedged() == 0);
        // Past the end of the file: the read fails and its result is nullptr.
        const HedgedReads::Request bad{0, bytes.size() - 10, 100};
        const uint8_t* out = &bytes[0];
        reads.read_all(std::span(&bad, 1), std::span(&out, 1));
        CHECK(out == nullptr);
        reads.release();
    }
    std::free(memory);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: hedged_reads_test DIR\n");
        return 2;
    }
    std::vector<uint8_t> bytes(64 * kPage);
    for (size_t i = 0; i < bytes.size(); ++i)
        bytes[i] = static_cast<uint8_t>(i * 7 % 251);
    const std::filesystem::path path = std::filesystem::path(argv[1]) / "hedged_reads_test.bin";
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), std::ssize(bytes));
    {
        const DirectFile f(path);
        check(f, bytes, std::chrono::seconds(10), false);     // never hedged
        check(f, bytes, std::chrono::microseconds(0), true);  // hedged almost always
    }
    std::filesystem::remove(path);
    return test::test_exit_code();
}
