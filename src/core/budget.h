// The memory budget (architecture decision 1): every reservation the engine makes at start, and the room left for
// routed experts above the free-memory floor. Planning is a pure function so it is unit-tested; the engine fills the
// room with a placement (core/placement.h), prints the table at start and refuses to start below the floor.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace torch {

inline constexpr uint64_t kGiB = uint64_t{1} << 30;
inline constexpr uint64_t kMiB = uint64_t{1} << 20;

// MemAvailable the engine leaves untouched: the Orin's GPU resets under memory pressure.
inline constexpr uint64_t kFreeFloor = 6 * kGiB;
// After loading, the engine fills MemAvailable down to the floor plus this margin with more experts: the reservations
// above are estimates, and loading also frees page cache that MemAvailable did not count (on the development Orin
// MemAvailable fell 37.5 GiB for a planned 45 GiB). With 1 GiB, runs ended at 6.9 GiB; the board is shared.
inline constexpr uint64_t kTopUpMargin = 3 * kGiB / 2;

struct Reservation {
    std::string name;
    uint64_t bytes;
};

struct Budget {
    uint64_t available = 0;  // MemAvailable when the engine started
    uint64_t floor = 0;
    std::vector<Reservation> reservations;  // everything except the resident experts

    // The placement, filled in once it is chosen.
    int resident = 0;             // experts resident, over all layers
    int experts = 0;              // experts in the model, over all layers
    uint64_t resident_bytes = 0;  // their memory
    std::string placement;        // how they were chosen
    // Experts added after loading, when MemAvailable was measured again.
    uint64_t available_after_load = 0;
    int added = 0;
    uint64_t added_bytes = 0;

    uint64_t reserved() const noexcept;                                        // sum of the reservations
    uint64_t room() const noexcept { return available - floor - reserved(); }  // for resident experts
    uint64_t left() const noexcept { return room() - resident_bytes; }
    std::string table() const;  // one line per item, as the engine prints it
};

// The reservations and the room they leave above the floor. Throws Error when the reservations alone do not fit.
Budget plan_budget(uint64_t available, uint64_t floor, std::vector<Reservation> reservations);

// The MemAvailable line of /proc/meminfo, in bytes. Throws Error when the text has none.
uint64_t parse_mem_available(std::string_view meminfo);

// MemAvailable now, from /proc/meminfo. Throws Error where that file does not exist (not Linux).
uint64_t mem_available();

}  // namespace torch
