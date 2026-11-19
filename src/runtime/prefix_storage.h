#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

struct CUstream_st;

namespace torch::runtime {

using DeviceRanges = std::vector<std::span<uint8_t>>;
uint64_t range_bytes(const DeviceRanges& parts);
void copy_device(const DeviceRanges& parts, uint8_t* packed, bool save, CUstream_st* stream);

struct PrefixFile {
    PrefixFile() = default;
    ~PrefixFile();
    PrefixFile(const PrefixFile&) = delete;
    PrefixFile& operator=(const PrefixFile&) = delete;
    int fd = -1;
    std::vector<uint32_t> checksums;
};

// Unnamed files stay private to one engine and disappear when its last file descriptor closes.
class PrefixDisk {
public:
    static constexpr uint64_t staging_bytes = 64ull << 20;
    PrefixDisk(const std::filesystem::path& directory, CUstream_st* stream);
    ~PrefixDisk();
    std::shared_ptr<PrefixFile> write(const DeviceRanges& parts);
    bool read(PrefixFile& file, const DeviceRanges& parts);

private:
    bool transfer(PrefixFile& file, const DeviceRanges& parts, bool save);
    int directory_ = -1;
    uint8_t* staging_ = nullptr;
    CUstream_st* stream_;
};

}  // namespace torch::runtime
