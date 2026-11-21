#include "runtime/prefix_cache.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <limits>
#include <optional>
#include <utility>

#include "core/cuda_check.h"
#include "core/error.h"
#include "runtime/direct_io.h"
#include "runtime/prefix_storage.h"

namespace torch::runtime {

void PrefixOptions::reserve(std::vector<Reservation>& out) const {
    if (memory_bytes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 2 ||
        disk_bytes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 2)
        throw Error("The prefix cache size is too large.");
    if (directory.empty() != (disk_bytes == 0))
        throw Error("Give a private prefix disk directory and a positive disk limit together.");
    if (memory_bytes)
        out.push_back({"prefix checkpoints in memory", memory_bytes});
    if (disk_bytes)
        out.push_back({"prefix disk transfer buffers", PrefixDisk::staging_bytes});
}

struct PrefixCache::Impl {
    struct Blob {
        uint64_t size = 0, at = 0;
        std::shared_ptr<PrefixFile> file;
        bool memory() const { return size && !file; }
    };
    struct Saved {
        Blob state, rows;
    };
    PrefixIndex index;
    std::map<PrefixKey, Saved> saved;
    model::Model& model;
    model::Draft* draft;
    int variants;
    uint64_t memory_cap, disk_cap;
    uint8_t* pool = nullptr;
    std::unique_ptr<PrefixDisk> disk;
    // Tests only (tests/model/checkpoints.py, tier "damaged"): the parked rows of the checkpoint this many tokens long
    // are damaged on disk just before a restore reads them, so the restore must fall back to a shorter prefix.
    size_t damage_at = 0;

    Impl(model::Model& m, model::Draft* d, int n, const PrefixOptions& o)
        : model(m), draft(d), variants(n), memory_cap(o.memory_bytes), disk_cap(o.disk_bytes) {
        if (disk_cap)
            disk = std::make_unique<PrefixDisk>(o.directory, m.stream());
        if (memory_cap)
            TORCH_CUDA_CHECK(cudaMalloc(&pool, memory_cap));
        if (const char* at = std::getenv("TORCH_TEST_DAMAGE_PREFIX_AT"))
            damage_at = std::strtoull(at, nullptr, 10);
    }
    ~Impl() {
        if (pool)
            cudaFree(pool);
    }

    DeviceRanges state(size_t pos) const {
        DeviceRanges out{model.checkpoint_state()};
        if (draft) {
            out.push_back(model.checkpoint_residual());
            draft->checkpoint_rows(static_cast<int>(pos), variants, out);
        }
        return out;
    }
    DeviceRanges rows(PrefixKey key) const {
        const auto& e = index.entries.at(key);
        const auto base = e.block ? index.entries.at(e.block).ids.size() : 0;
        DeviceRanges out;
        model.checkpoint_rows(static_cast<int>(base), static_cast<int>(e.ids.size()), out);
        return out;
    }

    void drop(PrefixKey key) {
        for (auto i = index.entries.begin(); i != index.entries.end();) {
            if (i->first == key || i->second.block == key) {
                saved.erase(i->first);
                i = index.entries.erase(i);
            } else
                ++i;
        }
        for (auto& [_, e] : index.entries)
            if (!index.entries.contains(e.previous))
                e.previous = 0;
    }

    bool protected_by(PrefixKey key, PrefixKey a, PrefixKey b) const {
        const auto protects = [&](PrefixKey id) {
            return id && (id == key || (index.entries.contains(id) && index.entries.at(id).block == key));
        };
        return protects(a) || protects(b);
    }
    std::optional<uint64_t> gap(uint64_t size) const {
        if (size > memory_cap)
            return {};
        std::vector<std::pair<uint64_t, uint64_t>> ranges;
        for (const auto& [_, s] : saved)
            for (const auto* b : {&s.state, &s.rows})
                if (b->memory())
                    ranges.emplace_back(b->at, b->size);
        std::sort(ranges.begin(), ranges.end());
        uint64_t at = 0;
        for (auto [start, n] : ranges) {
            if (start - at >= size)
                return at;
            at = start + n;
        }
        return memory_cap - at >= size ? std::optional<uint64_t>{at} : std::nullopt;
    }
    uint64_t disk_used() const {
        uint64_t bytes = 0;
        for (const auto& [_, s] : saved)
            for (const auto* b : {&s.state, &s.rows})
                if (b->file)
                    bytes += page_ceil(b->size);
        return bytes;
    }

    Blob on_disk(const DeviceRanges& regions, PrefixKey a, PrefixKey b, PrefixKey moving = 0) {
        const uint64_t size = range_bytes(regions), used = page_ceil(size);
        if (!disk || used > disk_cap)
            return {};
        while (disk_used() > disk_cap - used) {
            const auto victim = index.victim([&](PrefixKey k) {
                if (protected_by(k, a, b) || protected_by(k, moving, 0))
                    return false;
                const auto& s = saved.at(k);
                return s.state.file || s.rows.file;
            });
            if (!victim)
                return {};
            drop(victim);
        }
        auto f = disk->write(regions);
        return f ? Blob{size, 0, std::move(f)} : Blob{};
    }

    Blob store(const DeviceRanges& regions, PrefixKey a, PrefixKey b) {
        const uint64_t size = range_bytes(regions);
        if (size <= memory_cap) {
            for (;;) {
                if (auto at = gap(size)) {
                    copy_device(regions, pool + *at, true, model.stream());
                    return {size, *at, {}};
                }
                const auto victim = index.victim([&](PrefixKey k) {
                    if (protected_by(k, a, b))
                        return false;
                    const auto& s = saved.at(k);
                    return s.state.memory() || s.rows.memory();
                });
                if (!victim)
                    break;
                auto& s = saved.at(victim);
                bool moved = true;
                for (auto* blob : {&s.state, &s.rows}) {
                    if (!blob->memory())
                        continue;
                    auto next = on_disk({std::span(pool + blob->at, blob->size)}, a, b, victim);
                    if (!next.file) {
                        moved = false;
                        break;
                    }
                    *blob = std::move(next);
                }
                if (!moved)
                    drop(victim);
            }
        }
        return on_disk(regions, a, b);
    }

    bool read(const Blob& b, const DeviceRanges& regions) {
        if (range_bytes(regions) != b.size)
            return false;
        if (b.file)
            return disk->read(*b.file, regions);
        if (b.size)
            copy_device(regions, pool + b.at, false, model.stream());
        return true;
    }

    void park_others(PrefixKey target, PrefixStats& stats) {
        const std::span<const int32_t> retained =
            target ? std::span(index.entries.at(target).ids) : std::span<const int32_t>{};
        for (auto k : index.to_park(retained)) {
            if (!index.entries.contains(k))
                continue;
            auto& b = saved.at(k).rows;
            const auto regions = rows(k);
            const uint64_t bytes = range_bytes(regions);
            if (bytes && !b.size) {
                b = store(regions, target, k);
                if (!b.size) {
                    drop(k);
                    continue;
                }
                stats.parked_bytes += bytes;
            }
            index.entries.at(k).live = false;
        }
    }

    size_t resume(std::span<const int32_t> prompt, size_t stable, PrefixStats& stats) {
        for (;;) {
            const auto key = index.match(prompt, std::min(stable, prompt.size() - 1));
            park_others(key, stats);
            if (!key) {
                if (model.position())
                    model.reset();
                return 0;
            }
            const auto start = std::chrono::steady_clock::now();
            auto& e = index.entries.at(key);
            auto tier = PrefixTier::arena;
            bool failed = false;
            for (auto k : {e.block, key}) {
                if (!k || index.entries.at(k).live)
                    continue;
                const auto& b = saved.at(k).rows;
                tier = std::max(tier, b.file ? PrefixTier::disk : PrefixTier::memory);
                if (b.file && damage_at && index.entries.at(k).ids.size() == damage_at)
                    disk->damage(*b.file);
                if (!read(b, rows(k))) {
                    drop(k);
                    failed = true;
                    break;
                }
                index.entries.at(k).live = true;
            }
            if (failed) {
                continue;
            }
            const auto& b = saved.at(key).state;
            if (b.file)
                tier = PrefixTier::disk;
            if (!read(b, state(e.ids.size()))) {
                drop(key);
                continue;
            }
            model.restore_checkpoint(e.ids);
            if (draft) {
                std::vector<model::Drafted> ignored;
                for (int v = 0; v < variants; ++v)
                    draft->run(v, static_cast<int>(e.ids.size()) - 1, prompt.subspan(e.ids.size(), 1), 0, ignored);
            }
            for (auto k : {e.block, key})
                if (k) {
                    saved.at(k).rows = {};
                    index.touch(k);
                }
            stats.tier = tier;
            stats.restore_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            return e.ids.size();
        }
    }

    void save(std::span<const int32_t> ids, PrefixKind kind) {
        if (ids.empty() || ids.size() > static_cast<size_t>(model.checkpoint_limit()))
            return;
        PrefixKey previous = 0;
        for (const auto& [k, e] : index.entries) {
            if (!starts_with_ids(ids, e.ids))
                continue;
            if (e.kind == kind && e.ids.size() == ids.size())
                return;
            if (kind == PrefixKind::turn && e.kind == PrefixKind::turn && e.ids.size() < ids.size() &&
                (!previous || e.ids.size() > index.entries.at(previous).ids.size()))
                previous = k;
        }
        const auto root = kind == PrefixKind::turn ? index.match(ids, ids.size(), true) : 0;
        const auto old = index.old_turn(previous);
        const auto key = index.insert(ids, kind, root, previous);
        auto& data = saved[key];
        data.state = store(state(ids.size()), key, previous);
        if (!data.state.size) {
            drop(key);
            return;
        }
        if (old && index.entries.contains(old))
            drop(old);
    }
};

PrefixCache::PrefixCache(model::Model& m, model::Draft* d, int n, const PrefixOptions& o)
    : impl_(std::make_unique<Impl>(m, d, n, o)) {}
PrefixCache::~PrefixCache() = default;

size_t PrefixCache::resume(std::span<const int32_t> prompt, size_t stable, PrefixStats& stats) {
    return impl_->resume(prompt, stable, stats);
}
void PrefixCache::save(std::span<const int32_t> ids, PrefixKind kind) {
    impl_->save(ids, kind);
}
void PrefixCache::counts(PrefixStats& stats) const {
    for (const auto& [_, s] : impl_->saved) {
        if (s.state.file || s.rows.file)
            ++stats.on_disk;
        else
            ++stats.in_memory;
    }
}
void PrefixCache::clear() {
    impl_->saved.clear();
    impl_->index.entries.clear();
}

}  // namespace torch::runtime
