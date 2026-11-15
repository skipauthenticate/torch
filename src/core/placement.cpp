#include "core/placement.h"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>

#include "core/error.h"

namespace torch {
namespace {

constexpr const char* kMagic = "torch route counts";

}  // namespace

RouteCounts RouteCounts::read(const std::filesystem::path& path) {
    std::ifstream in(path);
    std::string magic;
    if (!in || !std::getline(in, magic) || magic != kMagic)
        throw Error("Cannot read route counts from " + path.string() +
                    "; pass a file written by torch run --route-counts.");
    RouteCounts c;
    if (!(in >> c.n_layer >> c.n_expert) || c.n_layer <= 0 || c.n_expert <= 0 || c.n_layer > 4096 || c.n_expert > 65536)
        throw Error("The route counts in " + path.string() + " have a bad shape line.");
    c.counts.resize(static_cast<size_t>(c.n_layer) * static_cast<size_t>(c.n_expert));
    for (uint64_t& v : c.counts) {
        if (!(in >> v))
            throw Error("The route counts in " + path.string() + " are cut short.");
    }
    return c;
}

void RouteCounts::write(const std::filesystem::path& path) const {
    std::ostringstream out;
    out << kMagic << "\n" << n_layer << " " << n_expert << "\n";
    for (int l = 0; l < n_layer; ++l) {
        for (int e = 0; e < n_expert; ++e)
            out << (e ? " " : "") << at(l, e);
        out << "\n";
    }
    std::ofstream file(path);
    if (!(file << out.str()))
        throw Error("Cannot write route counts to " + path.string() + ".");
}

void RouteCounts::add(const RouteCounts& other) {
    if (other.n_layer != n_layer || other.n_expert != n_expert)
        throw Error("Route counts of different models cannot be added.");
    for (size_t i = 0; i < counts.size(); ++i)
        counts[i] += other.counts[i];
}

int Placement::count() const {
    int n = 0;
    for (const std::vector<int32_t>& r : resident)
        n += static_cast<int>(r.size());
    return n;
}

Placement place_uniform(int k, int n_expert, const std::vector<uint64_t>& expert_bytes) {
    Placement p;
    k = std::clamp(k, 0, n_expert);
    for (uint64_t bytes : expert_bytes) {
        std::vector<int32_t> r(static_cast<size_t>(k));
        std::iota(r.begin(), r.end(), 0);
        p.resident.push_back(std::move(r));
        p.bytes += bytes * static_cast<uint64_t>(k);
    }
    return p;
}

int uniform_capacity(uint64_t room, int n_expert, const std::vector<uint64_t>& expert_bytes) {
    const uint64_t per_k = std::accumulate(expert_bytes.begin(), expert_bytes.end(), uint64_t{0});
    if (per_k == 0)
        return n_expert;
    return static_cast<int>(std::min<uint64_t>(room / per_k, static_cast<uint64_t>(n_expert)));
}

Placement place_ranked(const RouteCounts& counts, const std::vector<uint64_t>& expert_bytes, uint64_t room,
                       size_t limit) {
    return extend_ranked(counts, expert_bytes, Placement{}, room, limit);
}

Placement extend_ranked(const RouteCounts& counts, const std::vector<uint64_t>& expert_bytes, const Placement& have,
                        uint64_t room, size_t limit) {
    if (static_cast<int>(expert_bytes.size()) != counts.n_layer)
        throw Error("The route counts were recorded with another model; record them again with torch run.");
    struct Item {
        uint64_t count;
        int layer, expert;
    };
    std::vector<bool> resident(counts.counts.size(), false);
    for (size_t l = 0; l < have.resident.size(); ++l)
        for (const int32_t e : have.resident[l])
            resident[l * static_cast<size_t>(counts.n_expert) + static_cast<size_t>(e)] = true;
    std::vector<Item> items;
    items.reserve(counts.counts.size());
    for (int l = 0; l < counts.n_layer; ++l)
        for (int e = 0; e < counts.n_expert; ++e)
            if (!resident[static_cast<size_t>(l * counts.n_expert + e)])
                items.push_back({counts.at(l, e), l, e});
    std::stable_sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.count > b.count; });
    Placement p;
    p.resident.resize(static_cast<size_t>(counts.n_layer));
    size_t taken = 0;
    for (const Item& it : items) {
        const uint64_t bytes = expert_bytes[static_cast<size_t>(it.layer)];
        if (taken == limit)
            break;
        if (p.bytes + bytes > room)
            continue;
        ++taken;
        p.bytes += bytes;
        p.resident[static_cast<size_t>(it.layer)].push_back(it.expert);
    }
    for (std::vector<int32_t>& r : p.resident)
        std::sort(r.begin(), r.end());
    return p;
}

RouteCounts uniform_ranking(int n_layer, int n_expert) {
    RouteCounts c{n_layer, n_expert, {}};
    c.counts.resize(static_cast<size_t>(n_layer) * static_cast<size_t>(n_expert));
    for (int l = 0; l < n_layer; ++l)
        for (int e = 0; e < n_expert; ++e)
            c.counts[static_cast<size_t>(l * n_expert + e)] = static_cast<uint64_t>(n_expert - e);
    return c;
}

}  // namespace torch
