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
    char label[96];
    std::snprintf(label, sizeof label, "resident experts: %d of %d (%.1f%%)", resident, experts,
                  100.0 * resident / std::max(experts, 1));
    out += line(label, resident_bytes);
    if (!placement.empty())
        out += "    " + placement + "\n";
    out += line("free-memory floor", floor);
    out += line("left over", left());
    if (available_after_load > 0) {
        out += line("MemAvailable after loading", available_after_load);
        std::snprintf(label, sizeof label, "experts added to fill it to the floor + %.1f GiB: %d (%.1f%% resident)",
                      static_cast<double>(kTopUpMargin) / static_cast<double>(kGiB), added,
                      100.0 * (resident + added) / std::max(experts, 1));
        out += line(label, added_bytes);
    }
    if (tenants > 0) {
        std::snprintf(label, sizeof label, "KV cache lent from position %d (--lend-kv): %d experts", lend_from,
                      tenants);
        out += line(label, lent_bytes);
    }
    return out;
}

Budget plan_budget(uint64_t available, uint64_t floor, std::vector<Reservation> reservations) {
    Budget b;
    b.available = available;
    b.floor = floor;
    b.reservations = std::move(reservations);
    const uint64_t needed = floor + b.reserved();
    if (available < needed)
        throw Error("The engine needs " + gib(needed) + " of free memory before any expert is resident but only " +
                    gib(available) + " is available; stop other programs or shorten the context.");
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
