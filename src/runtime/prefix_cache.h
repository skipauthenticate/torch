// Saved prompt boundaries for one engine. The memory reservation is made before the model chooses its experts.
#pragma once

#include <filesystem>
#include <memory>

#include "core/prefix_index.h"
#include "model/draft.h"
#include "model/model.h"

namespace torch::runtime {

struct PrefixOptions {
    uint64_t memory_bytes = 0, disk_bytes = 0;
    std::filesystem::path directory;
    bool enabled() const { return memory_bytes || (!directory.empty() && disk_bytes); }
    void reserve(std::vector<Reservation>& reservations) const;
};

enum class PrefixTier { none, arena, memory, disk };
struct PrefixStats {
    PrefixTier tier = PrefixTier::none;
    double restore_seconds = 0;
    uint64_t parked_bytes = 0;
    int in_memory = 0, on_disk = 0;
};

class PrefixCache {
public:
    PrefixCache(model::Model& model, model::Draft* draft, int variants, const PrefixOptions& options);
    ~PrefixCache();
    size_t resume(std::span<const int32_t> prompt, size_t stable, PrefixStats& stats);
    void save(std::span<const int32_t> ids, PrefixKind kind);
    void counts(PrefixStats& stats) const;
    void clear();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace torch::runtime
