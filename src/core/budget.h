// The memory budget (architecture decision 1): every reservation the engine makes at start, and how many routed
// experts fit in what is left above the free-memory floor. Planning is a pure function so it is unit-tested; the
// engine prints the table at start and refuses to start below the floor.
//
// Until milestone M5 ranks experts, every layer keeps the same number of experts resident: experts 0 to k - 1 of
// each layer, so an expert's place is base + index * size and a miss is any index >= k.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace torch {

inline constexpr uint64_t kGiB = uint64_t{1} << 30;
inline constexpr uint64_t kMiB = uint64_t{1} << 20;

// MemAvailable the engine leaves untouched: the Orin's GPU resets under memory pressure.
inline constexpr uint64_t kFreeFloor = 6 * kGiB;

struct Reservation {
    std::string name;
    uint64_t bytes;
};

struct Budget {
    uint64_t available = 0;  // MemAvailable when the engine started
    uint64_t floor = 0;
    std::vector<Reservation> reservations;  // everything except the resident experts
    uint64_t expert_bytes = 0;              // one expert (gate, up and down) in every layer
    int experts_per_layer = 0;
    int resident = 0;  // experts resident in each layer

    uint64_t reserved() const noexcept;  // sum of the reservations
    uint64_t resident_bytes() const noexcept { return expert_bytes * static_cast<uint64_t>(resident); }
    uint64_t left() const noexcept { return available - floor - reserved() - resident_bytes(); }
    std::string table() const;  // one line per item, as the engine prints it
};

// Fits as many experts per layer as the memory left above the floor allows, or exactly `forced` of them. Throws Error
// when the reservations alone do not fit, or when `forced` experts do not.
Budget plan_budget(uint64_t available, uint64_t floor, std::vector<Reservation> reservations, uint64_t expert_bytes,
                   int experts_per_layer, std::optional<int> forced);

// The MemAvailable line of /proc/meminfo, in bytes. Throws Error when the text has none.
uint64_t parse_mem_available(std::string_view meminfo);

// MemAvailable now, from /proc/meminfo. Throws Error where that file does not exist (not Linux).
uint64_t mem_available();

}  // namespace torch
