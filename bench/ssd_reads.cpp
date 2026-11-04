// Latency and bandwidth of O_DIRECT reads from a model file at queue depths 1 to 16: 2 MiB reads (one expert
// matrix is 0.6-0.9 MiB, a whole expert 2.0-2.3 MiB) and 4 KiB reads (one PLE row lives in one page). Each queue
// slot is a thread issuing blocking preads at random 4 KiB-aligned offsets, so the page cache is never involved.
// A read is issued as preads of at most 128 KiB, as the engine reads (src/runtime/direct_io.cpp explains why); the
// worst column shows any read that stalled.
//
// usage: torch_bench_ssd FILE [MiB read per queue depth, default 1024]
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Result {
    double gb_per_s;
    double median_ms;
    double mean_ms;
    double p99_ms;
    double worst_ms;
};

constexpr size_t kPiece = size_t{128} << 10;

// `depth` threads each issue `per_thread` reads of `size` bytes; returns bandwidth and per-read latency.
Result run(int fd, uint64_t file_size, size_t size, int depth, int per_thread, uint64_t seed) {
    std::vector<std::vector<double>> latencies(depth);
    std::vector<std::thread> threads;
    bool failed = false;
    const auto start = Clock::now();
    for (int t = 0; t < depth; ++t) {
        threads.emplace_back([&, t] {
            void* buffer = std::aligned_alloc(4096, size);
            std::mt19937_64 rng(seed + t);
            const uint64_t pages = (file_size - size) / 4096;
            for (int i = 0; i < per_thread; ++i) {
                const off_t offset = static_cast<off_t>(rng() % pages * 4096);
                const auto begin = Clock::now();
                for (size_t done = 0; done < size; done += kPiece) {
                    const size_t n = std::min(kPiece, size - done);
                    if (pread(fd, static_cast<char*>(buffer) + done, n, offset + static_cast<off_t>(done)) !=
                        static_cast<ssize_t>(n))
                        failed = true;
                }
                latencies[t].push_back(std::chrono::duration<double, std::milli>(Clock::now() - begin).count());
            }
            std::free(buffer);
        });
    }
    for (std::thread& t : threads)
        t.join();
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    if (failed) {
        std::fprintf(stderr, "a read failed (%s)\n", std::strerror(errno));
        std::exit(1);
    }
    std::vector<double> all;
    for (const auto& l : latencies)
        all.insert(all.end(), l.begin(), l.end());
    std::sort(all.begin(), all.end());
    double sum = 0;
    for (double v : all)
        sum += v;
    const double n = static_cast<double>(all.size());
    return {static_cast<double>(size) * n / seconds / 1e9, all[all.size() / 2], sum / n, all[all.size() * 99 / 100],
            all.back()};
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: torch_bench_ssd FILE [MiB read per queue depth, default 1024]\n");
        return 2;
    }
    const double mib = argc > 2 ? std::atof(argv[2]) : 1024;
    const int fd = open(argv[1], O_RDONLY | O_DIRECT);
    if (fd < 0) {
        std::fprintf(stderr, "cannot open %s with O_DIRECT (%s)\n", argv[1], std::strerror(errno));
        return 1;
    }
    const uint64_t file_size = static_cast<uint64_t>(lseek(fd, 0, SEEK_END));
    std::printf("O_DIRECT random reads from %s (%.1f GiB)\n", argv[1], static_cast<double>(file_size) / 1073741824.0);
    std::printf("  size      depth      GB/s median ms   mean ms    p99 ms  worst ms\n");
    uint64_t seed = 1;
    for (const size_t size : {size_t{2} << 20, size_t{4096}}) {
        // 4 KiB reads are latency-bound: read a fixed count rather than a fixed volume
        const int total =
            size == 4096 ? 4096 : std::max(16, static_cast<int>(mib * (1 << 20) / static_cast<double>(size)));
        for (const int depth : {1, 2, 4, 8, 16}) {
            const Result r = run(fd, file_size, size, depth, std::max(1, total / depth), seed);
            seed += 100;
            std::printf("  %-8s %6d %9.2f %9.3f %9.3f %9.3f %9.1f\n", size == 4096 ? "4 KiB" : "2 MiB", depth,
                        r.gb_per_s, r.median_ms, r.mean_ms, r.p99_ms, r.worst_ms);
        }
    }
    close(fd);
    return 0;
}
