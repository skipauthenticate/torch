// Token prefixes and their dependencies. Only the engine decides which saved prefix a request can use.
#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <span>
#include <vector>

namespace torch {

enum class PrefixKind : uint8_t { block = 1, turn = 2 };
using PrefixKey = uint64_t;

struct PrefixEntry {
    std::vector<int32_t> ids;
    PrefixKind kind;
    PrefixKey block = 0, previous = 0;
    uint64_t used = 0;
    bool live = true;
};

inline bool starts_with_ids(std::span<const int32_t> ids, std::span<const int32_t> prefix) {
    return ids.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), ids.begin());
}

class PrefixIndex {
public:
    std::map<PrefixKey, PrefixEntry> entries;

    PrefixKey match(std::span<const int32_t> ids, size_t limit, bool blocks_only = false) const {
        PrefixKey best = 0;
        size_t length = 0;
        for (const auto& [key, e] : entries) {
            const size_t n = e.ids.size();
            if (n > limit || n < length || (blocks_only && e.kind != PrefixKind::block) ||
                (n == length && e.kind != PrefixKind::turn) || !starts_with_ids(ids, e.ids))
                continue;
            best = key;
            length = n;
        }
        return best;
    }

    // Rows below the restore point stay valid when their token ids match the retained prefix.
    std::vector<PrefixKey> to_park(std::span<const int32_t> retained) const {
        std::vector<PrefixKey> keys;
        for (const auto& [key, e] : entries)
            if (e.live && !starts_with_ids(retained, e.ids))
                keys.push_back(key);
        return keys;
    }

    PrefixKey insert(std::span<const int32_t> ids, PrefixKind kind, PrefixKey block, PrefixKey previous) {
        const auto key = ++clock_;
        entries.emplace(key, PrefixEntry{{ids.begin(), ids.end()}, kind, block, previous, key, true});
        return key;
    }

    void touch(PrefixKey key) { entries.at(key).used = ++clock_; }

    // An edited turn can share its predecessor with another branch of the same conversation.
    PrefixKey old_turn(PrefixKey previous) const {
        if (!previous || !entries.contains(previous))
            return 0;
        const auto older = entries.at(previous).previous;
        if (older && std::count_if(entries.begin(), entries.end(),
                                   [older](const auto& pair) { return pair.second.previous == older; }) == 1)
            return older;
        return 0;
    }

    template <class Eligible>
    PrefixKey victim(Eligible eligible) const {
        PrefixKey best = 0;
        for (const auto& [key, e] : entries) {
            if (!eligible(key))
                continue;
            if (!best ||
                (e.kind != entries.at(best).kind ? e.kind == PrefixKind::turn : e.used < entries.at(best).used))
                best = key;
        }
        return best;
    }

private:
    uint64_t clock_ = 0;
};

}  // namespace torch
