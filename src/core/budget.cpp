#include "core/budget.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <fstream>
#include <iterator>

#include "core/error.h"

namespace torch {
namespace {

std::string gib(uint64_t bytes) {
    char text[32];
    std::snprintf(text, sizeof text, "%8.2f GiB", static_cast<double>(bytes) / static_cast<double>(kGiB));
    return text;
}

std::string line(std::string_view name, uint64_t bytes) {
    std::string out = "  ";
    out += name;
    out.append(name.size() < 52 ? 52 - name.size() : 1, ' ');
    return out + gib(bytes) + "\n";
}

}  // namespace

uint64_t Budget::reserved() const noexcept {
    uint64_t sum = 0;
    for (const Reservation& r : reservations)
        sum += r.bytes;
    return sum;
}

std::string Budget::table() const {
    std::string out = "memory budget\n" + line("MemAvailable at start", available);
    for (const Reservation& r : reservations)
        out += line(r.name, r.bytes);
    char experts[96];
    std::snprintf(experts, sizeof experts, "resident experts: %d of %d per layer (%.1f%%)", resident, experts_per_layer,
                  100.0 * resident / std::max(experts_per_layer, 1));
    out += line(experts, resident_bytes());
    out += line("free-memory floor", floor);
    out += line("left over", left());
    return out;
}

Budget plan_budget(uint64_t available, uint64_t floor, std::vector<Reservation> reservations, uint64_t expert_bytes,
                   int experts_per_layer, std::optional<int> forced) {
    Budget b;
    b.available = available;
    b.floor = floor;
    b.reservations = std::move(reservations);
    b.expert_bytes = expert_bytes;
    b.experts_per_layer = experts_per_layer;
    const uint64_t needed = floor + b.reserved();
    if (available < needed)
        throw Error("The engine needs " + gib(needed) + " of free memory before any expert is resident but only " +
                    gib(available) + " is available; stop other programs or shorten the context.");
    const uint64_t room = available - needed;
    const uint64_t fit = expert_bytes == 0 ? static_cast<uint64_t>(experts_per_layer) : room / expert_bytes;
    const int most = static_cast<int>(std::min<uint64_t>(fit, static_cast<uint64_t>(experts_per_layer)));
    if (forced) {
        if (*forced < 0 || *forced > experts_per_layer)
            throw Error("--resident must be between 0 and " + std::to_string(experts_per_layer) + ".");
        if (*forced > most)
            throw Error("Only " + std::to_string(most) + " experts per layer fit in free memory, not " +
                        std::to_string(*forced) + "; ask for fewer with --resident.");
    }
    b.resident = forced.value_or(most);
    return b;
}

uint64_t parse_mem_available(std::string_view meminfo) {
    constexpr std::string_view kKey = "MemAvailable:";
    const size_t at = meminfo.find(kKey);
    if (at == std::string_view::npos)
        throw Error("/proc/meminfo has no MemAvailable line; the engine needs Linux 3.14 or newer.");
    const char* p = meminfo.data() + at + kKey.size();
    const char* end = meminfo.data() + meminfo.size();
    while (p < end && *p == ' ')
        ++p;
    uint64_t kib = 0;
    const auto [stop, error] = std::from_chars(p, end, kib);
    if (error != std::errc{} || std::string_view(stop, static_cast<size_t>(end - stop)).substr(0, 3) != " kB")
        throw Error("Cannot read the MemAvailable line of /proc/meminfo; check that it is in kB.");
    return kib * 1024;
}

uint64_t mem_available() {
    std::ifstream file("/proc/meminfo");
    if (!file)
        throw Error("Cannot read /proc/meminfo; the engine runs on Linux (JetPack) only.");
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return parse_mem_available(text);
}

}  // namespace torch
