// The window expert gate (src/runtime/expert_gate.h) with every expert resident: for windows of 1 to 8 columns with
// repeated experts across columns, the unique experts in order of first appearance, their column masks and ranks, the
// pointer table, the per-route down matrices, the count, the hit counter and the route counts, against a CPU
// computation. A one-column window must number its experts by rank. (Misses go through the host's I/O thread; the
// model checks cover them by running at residency 0.)
#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "../unit/check.h"
#include "device.h"
#include "runtime/expert_gate.h"

using namespace torch;
using runtime::kGateColumns;
using runtime::kGateSlots;
using test::DeviceBuffer;

namespace {

constexpr int kLayers = 2, kExperts = 64, kRoutes = 10;

// A fake address for (layer, expert, kind): never dereferenced.
const uint8_t* address(int layer, int expert, int kind) {
    return reinterpret_cast<const uint8_t*>(uintptr_t{0x100000} +
                                            uintptr_t(((layer * kExperts + expert) * 3 + kind) * 64));
}

bool check_window(int columns, uint32_t seed, const uint8_t* const* dmap) {
    std::mt19937 rng(seed);
    std::vector<int32_t> ids(static_cast<size_t>(columns * kRoutes));
    for (int t = 0; t < columns; ++t) {  // distinct within a column, overlapping across columns (16 of 64 experts)
        std::vector<int32_t> pool(16);
        for (int i = 0; i < 16; ++i)
            pool[static_cast<size_t>(i)] = (i * 4 + static_cast<int>(seed)) % kExperts;
        std::shuffle(pool.begin(), pool.end(), rng);
        for (int r = 0; r < kRoutes; ++r)
            ids[static_cast<size_t>(t * kRoutes + r)] = pool[static_cast<size_t>(r)];
    }
    const int layer = 1;
    DeviceBuffer<int32_t> dids(ids);
    DeviceBuffer<const uint8_t*> table(3 * kGateSlots), down(kGateColumns * kRoutes), gate(kGateColumns * kRoutes),
        up(kGateColumns * kRoutes);
    DeviceBuffer<uint8_t> masks(kGateSlots);
    DeviceBuffer<int8_t> ranks(kGateSlots * kGateColumns);
    DeviceBuffer<int32_t> count(1);
    DeviceBuffer<unsigned long long> counters(std::vector<unsigned long long>(2, 0));
    DeviceBuffer<unsigned long long> route_counts(std::vector<unsigned long long>(kLayers * kExperts, 0));
    DeviceBuffer<uint32_t> seq(std::vector<uint32_t>{0});
    DeviceBuffer<uint8_t> zeros(64);
    DeviceBuffer<int32_t> pos(std::vector<int32_t>{41});  // the window's first position: routes are stamped 42
    DeviceBuffer<uint32_t> last_use(std::vector<uint32_t>(kLayers * kExperts, 0));
    runtime::Mailbox* box = nullptr;  // unused: no misses
    TORCH_CUDA_CHECK(cudaMallocManaged(&box, sizeof(runtime::Mailbox)));
    *box = runtime::Mailbox{};
    const runtime::GateTables out{table.get(), masks.get(), ranks.get(), count.get(), down.get(), gate.get(), up.get()};
    runtime::expert_gate(dids.get(), kRoutes, columns, layer, dmap, kExperts, out, box, seq.get(), counters.get(),
                         route_counts.get(), zeros.get(), pos.get(), last_use.get(), nullptr);
    TORCH_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> unique;
    std::vector<uint8_t> want_mask(kGateSlots, 0);
    std::vector<int8_t> want_rank(kGateSlots * kGateColumns, -1);
    std::vector<int> slot_of(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        const auto it = std::find(unique.begin(), unique.end(), ids[i]);
        const int u = static_cast<int>(it - unique.begin());
        if (it == unique.end())
            unique.push_back(ids[i]);
        slot_of[i] = u;
        const int t = static_cast<int>(i) / kRoutes;
        want_mask[static_cast<size_t>(u)] |= static_cast<uint8_t>(1u << t);
        want_rank[static_cast<size_t>(u * kGateColumns + t)] = static_cast<int8_t>(static_cast<int>(i) % kRoutes);
    }
    bool ok = count.download()[0] == static_cast<int32_t>(unique.size());
    const std::vector<const uint8_t*> got_table = table.download(), got_down = down.download(),
                                      got_gate = gate.download(), got_up = up.download();
    const std::vector<uint8_t> got_mask = masks.download();
    const std::vector<int8_t> got_rank = ranks.download();
    for (size_t u = 0; u < unique.size(); ++u) {
        for (int kind = 0; kind < 3; ++kind)
            ok = ok && got_table[static_cast<size_t>(kind) * kGateSlots + u] == address(layer, unique[u], kind);
        ok = ok && got_mask[u] == want_mask[u];
    }
    ok = ok && got_rank == want_rank;
    for (size_t i = 0; i < ids.size(); ++i)
        ok = ok && got_down[i] == address(layer, unique[static_cast<size_t>(slot_of[i])], 2) &&
             got_gate[i] == address(layer, unique[static_cast<size_t>(slot_of[i])], 0) &&
             got_up[i] == address(layer, unique[static_cast<size_t>(slot_of[i])], 1);
    if (columns == 1)
        for (int r = 0; r < kRoutes; ++r)
            ok = ok && unique[static_cast<size_t>(r)] == ids[static_cast<size_t>(r)];
    const std::vector<unsigned long long> c = counters.download(), rc = route_counts.download();
    ok = ok && c[0] == ids.size() && c[1] == 0;
    const std::vector<uint32_t> stamps = last_use.download();
    for (int e = 0; e < kExperts; ++e) {
        const bool routed = std::count(ids.begin(), ids.end(), e) > 0;
        ok = ok && rc[static_cast<size_t>(layer * kExperts + e)] ==
                       static_cast<unsigned long long>(std::count(ids.begin(), ids.end(), e));
        ok = ok && stamps[static_cast<size_t>(layer * kExperts + e)] == (routed ? 42u : 0u) &&
             stamps[static_cast<size_t>(e)] == 0u;  // layer 0 untouched
    }
    cudaFree(box);
    std::printf("expert_gate %d columns (%zu unique of %zu routes)  %s\n", columns, unique.size(), ids.size(),
                ok ? "ok" : "FAIL");
    return ok;
}

// copy_bytes at every offset modulo 4 and odd lengths (an expert's matrices start anywhere within a page): the bytes
// are a host copy's, and the bytes around them untouched.
bool check_copy() {
    bool ok = true;
    for (const uint64_t at : {0u, 1u, 2u, 3u})
        for (const uint64_t n : {1ull, 3ull, 7ull, 1000003ull}) {
            std::vector<uint8_t> src(n + 16), dst(n + 16, 0xA5);
            for (size_t i = 0; i < src.size(); ++i)
                src[i] = static_cast<uint8_t>(i * 131 + 7);
            DeviceBuffer<uint8_t> ds(src), dd(dst);
            runtime::copy_bytes(dd.get() + at, ds.get() + at, n, nullptr);
            const std::vector<uint8_t> got = dd.download();
            for (size_t i = 0; i < got.size(); ++i)
                ok = ok && got[i] == (i >= at && i < at + n ? src[i] : 0xA5);
        }
    std::printf("copy_bytes at offsets 0-3, lengths 1 to 1000003  %s\n", ok ? "ok" : "FAIL");
    return ok;
}

}  // namespace

int main() {
    try {
        std::vector<const uint8_t*> map(static_cast<size_t>(kLayers * kExperts * 3));
        for (int l = 0; l < kLayers; ++l)
            for (int e = 0; e < kExperts; ++e)
                for (int kind = 0; kind < 3; ++kind)
                    map[static_cast<size_t>((l * kExperts + e) * 3 + kind)] = address(l, e, kind);
        DeviceBuffer<const uint8_t*> dmap(map);
        for (int columns = 1; columns <= kGateColumns; ++columns)
            CHECK(check_window(columns, static_cast<uint32_t>(columns), dmap.get()));
        CHECK(check_copy());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gate_test: %s\n", e.what());
        return 1;
    }
    return test::test_exit_code();
}
