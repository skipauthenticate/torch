// GGUF reader: parses one file's header (metadata and tensor table) and reads tensor bytes with pread. A model stored
// as several shard files opens as one File (File::open_model).
//
// Every count, length, dimension and offset in a header is checked before use; a malformed file throws torch::Error
// and is never read out of bounds. The whole file is mapped read-only so the header can be parsed in place, but the
// engine reads weights with read() into its own memory: only header pages are touched through the mapping.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace torch::gguf {

// Metadata value types as stored in GGUF.
enum class ValueType : uint32_t { U8 = 0, I8, U16, I16, U32, I32, F32, Bool, String, Array, U64, I64, F64 };

// "u8", "i32", "string", "array" and so on; "?" for an unknown type.
std::string_view value_type_name(ValueType type) noexcept;
// Bytes of one scalar; 0 for String, Array and unknown types.
size_t scalar_size(ValueType type) noexcept;

// One metadata value: a view into header bytes owned by the File.
struct Value {
    ValueType type;
    ValueType element;     // element type of an Array
    uint64_t count;        // bytes of a String, elements of an Array
    const uint8_t* bytes;  // the scalar, the string's bytes, or the first element (strings in arrays are u64-prefixed)
};

struct TensorInfo {
    std::string name;
    uint32_t type;                 // tensor type id; type_info(type) is never nullptr
    uint32_t n_dims;               // 1 to 4
    std::array<int64_t, 4> shape;  // shape[0] is the contiguous dimension; unused dimensions are 1
    uint64_t offset;               // from the start of the data section (open_model: the position in its parts)
    uint64_t nbytes;

    int64_t elements() const noexcept;
    uint64_t row_bytes() const noexcept;  // bytes of one shape[0] row
    // Bytes of one index along the last dimension (one expert of an "_exps" tensor), and where index starts,
    // relative to the tensor's first byte.
    uint64_t slice_bytes() const noexcept;
    uint64_t slice_offset(int64_t index) const noexcept;
};

// The split.* keys of one shard of a model stored as several files.
struct Split {
    uint16_t index;          // split.no, from 0
    uint16_t count;          // split.count
    uint64_t total_tensors;  // split.tensors.count, over all shards
};

// One file of a model as File::open_model lays them out: the file's byte i is the model's byte base + i.
struct Part {
    std::filesystem::path path;
    uint64_t base;  // a multiple of kPartAlignment
    uint64_t size;  // bytes of the file
};

// Parts start on 4 KiB boundaries, so a direct read of any part stays page-aligned.
inline constexpr uint64_t kPartAlignment = 4096;
// The most shards File::open_model accepts.
inline constexpr int kMaxShards = 64;

class File {
public:
    // Parses a header held in memory (tests, fuzzing). Tensors may extend past the end of the buffer; read() is not
    // available. Throws Error.
    static File parse(std::vector<uint8_t> header, std::string origin = "<memory>");
    // Maps the file read-only and parses its header. Every tensor must lie inside the file unless header_only is set:
    // then the file may stop anywhere after the tensor table, e.g. the first 11 MiB of shard 1 fetched with an HTTP
    // range request. Throws Error.
    static File open(const std::filesystem::path& path, bool header_only = false);
    // Opens a model: a single file, or shard 1 of a split one (NAME-00001-of-0000N.gguf, N <= kMaxShards), whose other
    // shards must be next to it under the same name (the path as given; links are not resolved). The shards' split.*
    // keys must agree, and together they must hold split.tensors.count tensors with distinct names. A split model
    // reads like one file: the metadata of shard 1, every shard's tensors in shard order, data_offset() 0 and each
    // tensor's offset its position in the shards laid end to end (parts(), the layout DirectFile reads); read() and
    // mapped() refuse it, since its bytes are in other files. header_only applies to every shard. Throws Error.
    static File open_model(const std::filesystem::path& path, bool header_only = false);

    uint32_t version() const noexcept { return version_; }          // always 3
    uint64_t alignment() const noexcept { return alignment_; }      // general.alignment, 32 when absent
    uint64_t data_offset() const noexcept { return data_offset_; }  // where the data section starts
    uint64_t file_size() const noexcept { return size_; }           // bytes of the file, or of parse()'s buffer
    std::optional<Split> split() const noexcept { return split_; }
    // The files behind this one: itself at base 0, or a model's shards; none for parse().
    std::span<const Part> parts() const noexcept { return parts_; }

    const Value* find(std::string_view key) const noexcept;
    // Typed access; nullopt when the key is missing or its type does not match. An integral T accepts any integer
    // type whose value fits in T; bool accepts Bool; float and double accept F32 and F64; std::string accepts String;
    // std::vector<T> accepts an Array whose elements T accepts. gguf.cpp lists the supported T.
    template <class T>
    std::optional<T> get(std::string_view key) const;
    std::span<const std::pair<std::string, Value>> metadata() const noexcept { return metadata_; }  // in file order

    std::span<const TensorInfo> tensors() const noexcept { return tensors_; }  // in file order
    const TensorInfo* tensor(std::string_view name) const noexcept;

    // Copies bytes [from, from + n) of a tensor into dst with pread. Throws Error when the range is outside the
    // tensor or the file (a header-only file that stops early), or when the read fails.
    void read(const TensorInfo& t, uint64_t from, uint64_t n, void* dst) const;
    // A view of a tensor's bytes through the mapping, for tests and inspection only: the engine never maps weights.
    // Throws Error when the bytes are not in the file or the buffer.
    std::span<const uint8_t> mapped(const TensorInfo& t) const;

private:
    // The open file and its read-only mapping; empty for parse(). Move-only.
    struct Mapping {
        int fd = -1;
        void* base = nullptr;
        size_t size = 0;

        Mapping() = default;
        Mapping(Mapping&& other) noexcept;
        Mapping& operator=(Mapping&& other) noexcept;
        ~Mapping();
    };

    File() = default;
    void parse_header(bool check_bounds);
    bool contains(uint64_t start, uint64_t n) const noexcept;  // [start, start + n) of the data section is present

    std::string origin_;             // the path, for error messages
    std::vector<uint8_t> buffer_;    // parse(): the header bytes
    Mapping mapping_;                // open(): the file
    const uint8_t* data_ = nullptr;  // the buffer or the mapping
    uint64_t size_ = 0;

    uint32_t version_ = 0;
    uint64_t alignment_ = 32;
    uint64_t data_offset_ = 0;
    std::optional<Split> split_;
    std::vector<std::pair<std::string, Value>> metadata_;
    std::map<std::string, size_t, std::less<>> keys_;  // metadata_ index by key
    std::vector<TensorInfo> tensors_;
    std::map<std::string, size_t, std::less<>> names_;  // tensors_ index by name
    std::vector<Part> parts_;  // open(): this file; open_model(): every shard, this one's mapping being shard 1's
};

}  // namespace torch::gguf
