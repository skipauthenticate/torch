#include "gguf/gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstring>
#include <iterator>
#include <type_traits>

#include "core/error.h"
#include "formats/types.h"

namespace torch::gguf {
namespace {

constexpr uint64_t kMaxString = uint64_t{1} << 30;  // far longer than any real key, name or token
constexpr int64_t kMaxDimension = int64_t{1} << 40;
constexpr uint64_t kMaxTensorBytes = uint64_t{1} << 50;  // so no size or element count can overflow
constexpr uint64_t kMaxRead = uint64_t{1} << 30;

template <class T>
T load(const uint8_t* p) {
    T value;
    std::memcpy(&value, p, sizeof value);
    return value;
}

[[noreturn]] void damaged(const std::string& origin, const std::string& problem) {
    throw Error(origin + " is not a valid GGUF file (" + problem + "); download it again.");
}

// Reads the header front to back; a read past the end of the bytes throws.
class Cursor {
public:
    Cursor(const uint8_t* data, uint64_t size, const std::string& origin) : data_(data), size_(size), origin_(origin) {}

    uint64_t position() const { return pos_; }
    uint64_t remaining() const { return size_ - pos_; }

    const uint8_t* take(uint64_t n) {
        if (n > remaining())
            damaged(origin_, "the header is truncated at byte " + std::to_string(size_));
        const uint8_t* p = data_ + pos_;
        pos_ += n;
        return p;
    }
    template <class T>
    T read() {
        return load<T>(take(sizeof(T)));
    }
    std::string_view string() {
        const uint64_t n = read<uint64_t>();
        if (n > kMaxString)
            damaged(origin_, "a string of " + std::to_string(n) + " bytes at byte " + std::to_string(pos_));
        return {reinterpret_cast<const char*>(take(n)), n};
    }

private:
    const uint8_t* data_;
    uint64_t size_;
    uint64_t pos_ = 0;
    const std::string& origin_;
};

Value read_value(Cursor& c, uint32_t type, const std::string& origin) {
    Value v{static_cast<ValueType>(type), ValueType::U8, 0, nullptr};
    if (v.type == ValueType::String) {
        const std::string_view s = c.string();
        v.count = s.size();
        v.bytes = reinterpret_cast<const uint8_t*>(s.data());
        return v;
    }
    if (v.type != ValueType::Array) {
        const size_t size = scalar_size(v.type);
        if (size == 0)
            damaged(origin, "unknown value type " + std::to_string(type));
        v.bytes = c.take(size);
        return v;
    }
    v.element = static_cast<ValueType>(c.read<uint32_t>());
    v.count = c.read<uint64_t>();
    v.bytes = c.take(0);
    if (v.element == ValueType::String) {
        for (uint64_t i = 0; i < v.count; ++i)
            c.string();  // each takes at least 8 bytes, so this ends
        return v;
    }
    const size_t size = scalar_size(v.element);
    if (v.element == ValueType::Array)
        damaged(origin, "a nested array");
    if (size == 0)
        damaged(origin, "an array of unknown value type " + std::to_string(static_cast<uint32_t>(v.element)));
    if (v.count > c.remaining() / size)
        damaged(origin, "an array runs past the end of the header");
    c.take(v.count * size);
    return v;
}

template <class T, class V>
std::optional<T> fit(V value) {
    if (!std::in_range<T>(value))
        return std::nullopt;
    return static_cast<T>(value);
}

template <class T>
std::optional<T> scalar(ValueType type, const uint8_t* p) {
    if constexpr (std::is_same_v<T, bool>) {
        if (type != ValueType::Bool)
            return std::nullopt;
        return *p != 0;
    } else if constexpr (std::is_floating_point_v<T>) {
        if (type == ValueType::F32)
            return static_cast<T>(load<float>(p));
        if (type == ValueType::F64)
            return static_cast<T>(load<double>(p));
        return std::nullopt;
    } else {
        switch (type) {
            case ValueType::U8:
                return fit<T>(load<uint8_t>(p));
            case ValueType::I8:
                return fit<T>(load<int8_t>(p));
            case ValueType::U16:
                return fit<T>(load<uint16_t>(p));
            case ValueType::I16:
                return fit<T>(load<int16_t>(p));
            case ValueType::U32:
                return fit<T>(load<uint32_t>(p));
            case ValueType::I32:
                return fit<T>(load<int32_t>(p));
            case ValueType::U64:
                return fit<T>(load<uint64_t>(p));
            case ValueType::I64:
                return fit<T>(load<int64_t>(p));
            default:
                return std::nullopt;
        }
    }
}

template <class T>
constexpr bool is_vector = false;
template <class T>
constexpr bool is_vector<std::vector<T>> = true;

}  // namespace

std::string_view value_type_name(ValueType type) noexcept {
    constexpr std::string_view kNames[] = {"u8",   "i8",     "u16",   "i16", "u32", "i32", "f32",
                                           "bool", "string", "array", "u64", "i64", "f64"};
    const auto i = static_cast<uint32_t>(type);
    return i < std::size(kNames) ? kNames[i] : "?";
}

size_t scalar_size(ValueType type) noexcept {
    switch (type) {
        case ValueType::U8:
        case ValueType::I8:
        case ValueType::Bool:
            return 1;
        case ValueType::U16:
        case ValueType::I16:
            return 2;
        case ValueType::U32:
        case ValueType::I32:
        case ValueType::F32:
            return 4;
        case ValueType::U64:
        case ValueType::I64:
        case ValueType::F64:
            return 8;
        case ValueType::String:
        case ValueType::Array:
            return 0;
    }
    return 0;
}

int64_t TensorInfo::elements() const noexcept {
    return shape[0] * shape[1] * shape[2] * shape[3];
}

uint64_t TensorInfo::row_bytes() const noexcept {
    return torch::row_bytes(*type_info(type), shape[0]);
}

uint64_t TensorInfo::slice_bytes() const noexcept {
    return nbytes / shape[n_dims - 1];
}

uint64_t TensorInfo::slice_offset(int64_t index) const noexcept {
    return index * slice_bytes();
}

File::Mapping::Mapping(Mapping&& other) noexcept
    : fd(std::exchange(other.fd, -1)), base(std::exchange(other.base, nullptr)), size(std::exchange(other.size, 0)) {}

File::Mapping& File::Mapping::operator=(Mapping&& other) noexcept {
    std::swap(fd, other.fd);  // other's destructor releases what this held
    std::swap(base, other.base);
    std::swap(size, other.size);
    return *this;
}

File::Mapping::~Mapping() {
    if (base != nullptr)
        munmap(base, size);
    if (fd >= 0)
        ::close(fd);
}

File File::parse(std::vector<uint8_t> header, std::string origin) {
    File f;
    f.origin_ = std::move(origin);
    f.buffer_ = std::move(header);
    f.data_ = f.buffer_.data();
    f.size_ = f.buffer_.size();
    f.parse_header(/*check_bounds=*/false);
    return f;
}

File File::open(const std::filesystem::path& path, bool header_only) {
    File f;
    f.origin_ = path.string();
    f.mapping_.fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    struct stat st{};
    if (f.mapping_.fd < 0 || fstat(f.mapping_.fd, &st) != 0) {
        const int error = errno;
        throw Error("Cannot open " + f.origin_ + " (" + std::strerror(error) + "); check the path.");
    }
    f.size_ = static_cast<uint64_t>(st.st_size);
    if (f.size_ > 0) {
        void* base = mmap(nullptr, f.size_, PROT_READ, MAP_PRIVATE, f.mapping_.fd, 0);
        if (base == MAP_FAILED) {
            const int error = errno;
            throw Error("Cannot map " + f.origin_ + " (" + std::strerror(error) +
                        "); check that it is a regular file.");
        }
        f.mapping_.base = base;
        f.mapping_.size = f.size_;
        f.data_ = static_cast<const uint8_t*>(base);
    }
    f.parse_header(/*check_bounds=*/!header_only);
    return f;
}

void File::parse_header(bool check_bounds) {
    if (size_ < 4 || std::memcmp(data_, "GGUF", 4) != 0) {
        throw Error(origin_ + " is not a GGUF file; check the path.");
    }
    Cursor c(data_, size_, origin_);
    c.take(4);
    version_ = c.read<uint32_t>();
    if (version_ != 3) {
        throw Error(origin_ + " is GGUF version " + std::to_string(version_) +
                    ", but only version 3 is supported; use the files listed in docs/model-files.md.");
    }
    const uint64_t n_tensors = c.read<uint64_t>();
    const uint64_t n_keys = c.read<uint64_t>();
    // Every entry takes at least 8 bytes, so a larger count cannot be right.
    if (n_tensors > size_ / 8 || n_keys > size_ / 8)
        damaged(origin_, "impossible tensor or key count");

    for (uint64_t i = 0; i < n_keys; ++i) {
        std::string key(c.string());
        const Value value = read_value(c, c.read<uint32_t>(), origin_);
        if (!keys_.emplace(key, metadata_.size()).second)
            damaged(origin_, "duplicate key " + key);
        metadata_.emplace_back(std::move(key), value);
    }

    if (find("general.alignment") != nullptr) {
        const std::optional<uint32_t> alignment = get<uint32_t>("general.alignment");
        if (!alignment || !std::has_single_bit(*alignment)) {
            damaged(origin_, "general.alignment is not a power of two");
        }
        alignment_ = *alignment;
    }
    if (find("split.count") != nullptr) {
        const auto index = get<uint16_t>("split.no");
        const auto count = get<uint16_t>("split.count");
        const auto total = get<uint64_t>("split.tensors.count");
        if (!index || !count || !total || *index >= *count)
            damaged(origin_, "inconsistent split.* keys");
        split_ = Split{*index, *count, *total};
    }

    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorInfo t{std::string(c.string()), 0, 0, {1, 1, 1, 1}, 0, 0};
        t.n_dims = c.read<uint32_t>();
        if (t.n_dims < 1 || t.n_dims > 4) {
            damaged(origin_, "tensor " + t.name + " has " + std::to_string(t.n_dims) + " dimensions");
        }
        for (uint32_t d = 0; d < t.n_dims; ++d) {
            t.shape[d] = c.read<int64_t>();
            if (t.shape[d] < 1 || t.shape[d] > kMaxDimension)
                damaged(origin_, "tensor " + t.name + " has a bad shape");
        }
        t.type = c.read<uint32_t>();
        t.offset = c.read<uint64_t>();

        const TypeInfo* info = type_info(t.type);
        if (info == nullptr)
            damaged(origin_, "tensor " + t.name + " has unknown type " + std::to_string(t.type));
        if (t.shape[0] % info->block_elements != 0) {
            damaged(origin_,
                    "tensor " + t.name + " has rows that are not whole " + std::string(info->name) + " blocks");
        }
        t.nbytes = torch::row_bytes(*info, t.shape[0]);
        for (uint32_t d = 1; d < t.n_dims; ++d) {
            if (t.nbytes > kMaxTensorBytes / t.shape[d])
                damaged(origin_, "tensor " + t.name + " is impossibly large");
            t.nbytes *= t.shape[d];
        }
        if (!names_.emplace(t.name, tensors_.size()).second)
            damaged(origin_, "duplicate tensor " + t.name);
        tensors_.push_back(std::move(t));
    }

    data_offset_ = (c.position() + alignment_ - 1) / alignment_ * alignment_;
    for (const TensorInfo& t : tensors_) {
        if (t.offset % alignment_ != 0)
            damaged(origin_, "tensor " + t.name + " is not aligned");
        if (check_bounds && !contains(t.offset, t.nbytes)) {
            throw Error(origin_ + " is incomplete (tensor " + t.name +
                        " ends after the end of the file); download it again.");
        }
    }
}

bool File::contains(uint64_t start, uint64_t n) const noexcept {
    const uint64_t available = size_ > data_offset_ ? size_ - data_offset_ : 0;
    return start <= available && n <= available - start;
}

const Value* File::find(std::string_view key) const noexcept {
    const auto it = keys_.find(key);
    return it == keys_.end() ? nullptr : &metadata_[it->second].second;
}

const TensorInfo* File::tensor(std::string_view name) const noexcept {
    const auto it = names_.find(name);
    return it == names_.end() ? nullptr : &tensors_[it->second];
}

template <class T>
std::optional<T> File::get(std::string_view key) const {
    const Value* v = find(key);
    if (v == nullptr)
        return std::nullopt;
    if constexpr (std::is_same_v<T, std::string>) {
        if (v->type != ValueType::String)
            return std::nullopt;
        return std::string(reinterpret_cast<const char*>(v->bytes), v->count);
    } else if constexpr (is_vector<T>) {
        using Element = typename T::value_type;
        if (v->type != ValueType::Array)
            return std::nullopt;
        T out;
        out.reserve(v->count);
        const uint8_t* p = v->bytes;
        if constexpr (std::is_same_v<Element, std::string>) {
            if (v->element != ValueType::String)
                return std::nullopt;
            for (uint64_t i = 0; i < v->count; ++i) {  // parse_header checked every length
                const uint64_t n = load<uint64_t>(p);
                out.emplace_back(reinterpret_cast<const char*>(p + 8), n);
                p += 8 + n;
            }
        } else {
            const size_t size = scalar_size(v->element);
            if (size == 0)
                return std::nullopt;
            for (uint64_t i = 0; i < v->count; ++i, p += size) {
                const std::optional<Element> element = scalar<Element>(v->element, p);
                if (!element)
                    return std::nullopt;
                out.push_back(*element);
            }
        }
        return out;
    } else {
        return scalar<T>(v->type, v->bytes);
    }
}

// The types get<T> supports.
template std::optional<bool> File::get<bool>(std::string_view) const;
template std::optional<uint16_t> File::get<uint16_t>(std::string_view) const;
template std::optional<uint32_t> File::get<uint32_t>(std::string_view) const;
template std::optional<uint64_t> File::get<uint64_t>(std::string_view) const;
template std::optional<int32_t> File::get<int32_t>(std::string_view) const;
template std::optional<int64_t> File::get<int64_t>(std::string_view) const;
template std::optional<float> File::get<float>(std::string_view) const;
template std::optional<double> File::get<double>(std::string_view) const;
template std::optional<std::string> File::get<std::string>(std::string_view) const;
template std::optional<std::vector<int32_t>> File::get<std::vector<int32_t>>(std::string_view) const;
template std::optional<std::vector<uint32_t>> File::get<std::vector<uint32_t>>(std::string_view) const;
template std::optional<std::vector<uint64_t>> File::get<std::vector<uint64_t>>(std::string_view) const;
template std::optional<std::vector<float>> File::get<std::vector<float>>(std::string_view) const;
template std::optional<std::vector<std::string>> File::get<std::vector<std::string>>(std::string_view) const;

void File::read(const TensorInfo& t, uint64_t from, uint64_t n, void* dst) const {
    if (mapping_.fd < 0)
        throw Error(origin_ + " was parsed from memory and has no file to read; use File::open.");
    if (from > t.nbytes || n > t.nbytes - from || t.offset > UINT64_MAX - from || !contains(t.offset + from, n)) {
        throw Error("Bytes [" + std::to_string(from) + ", +" + std::to_string(n) + ") of tensor " + t.name +
                    " are not in " + origin_ + "; open the complete file.");
    }
    auto* out = static_cast<uint8_t*>(dst);
    uint64_t pos = data_offset_ + t.offset + from;
    while (n > 0) {
        // One call reads at most 1 GiB: macOS rejects reads above 2 GiB and Linux shortens them.
        const ssize_t got = ::pread(mapping_.fd, out, std::min(n, kMaxRead), static_cast<off_t>(pos));
        const int error = errno;
        if (got < 0 && error == EINTR)
            continue;
        if (got <= 0) {
            throw Error("Cannot read " + origin_ + " at byte " + std::to_string(pos) + " (" +
                        (got < 0 ? std::strerror(error) : "unexpected end of file") + "); check the disk.");
        }
        out += got;
        pos += static_cast<uint64_t>(got);
        n -= static_cast<uint64_t>(got);
    }
}

std::span<const uint8_t> File::mapped(const TensorInfo& t) const {
    if (!contains(t.offset, t.nbytes)) {
        throw Error("Tensor " + t.name + " is not in " + origin_ + "; open the complete file.");
    }
    return {data_ + data_offset_ + t.offset, t.nbytes};
}

}  // namespace torch::gguf
