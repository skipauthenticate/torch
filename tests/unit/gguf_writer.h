// Writes GGUF headers field by field (for the malformed files tests/fixtures/gguf_writer.py cannot write), and a small
// model split into shard files in either published layout: weights in shard 1 and the rest in shard 2, or shard 1
// with metadata only and the weights in the others.
#pragma once

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "formats/types.h"
#include "gguf/gguf.h"

namespace torch::test {

struct Writer {
    std::vector<uint8_t> bytes;

    template <class T>
    Writer& put(T value) {
        const auto* p = reinterpret_cast<const uint8_t*>(&value);
        bytes.insert(bytes.end(), p, p + sizeof value);
        return *this;
    }
    Writer& string(std::string_view s) {
        put<uint64_t>(s.size());
        bytes.insert(bytes.end(), s.begin(), s.end());
        return *this;
    }
    Writer& header(uint64_t n_tensors, uint64_t n_keys) {
        bytes.insert(bytes.end(), {'G', 'G', 'U', 'F'});
        return put<uint32_t>(3).put(n_tensors).put(n_keys);
    }
    Writer& key_u32(std::string_view key, uint32_t value) { return string(key).put(gguf::ValueType::U32).put(value); }
    Writer& key_u16(std::string_view key, uint16_t value) { return string(key).put(gguf::ValueType::U16).put(value); }
    Writer& key_i32(std::string_view key, int32_t value) { return string(key).put(gguf::ValueType::I32).put(value); }
    Writer& key_string(std::string_view key, std::string_view value) {
        return string(key).put(gguf::ValueType::String).string(value);
    }
    Writer& tensor(std::string_view name, std::initializer_list<int64_t> shape, uint32_t type, uint64_t offset) {
        string(name).put(static_cast<uint32_t>(shape.size()));
        for (int64_t d : shape)
            put(d);
        return put(type).put(offset);
    }
    Writer& split(uint16_t index, uint16_t count, int32_t total) {
        return key_u16("split.no", index).key_u16("split.count", count).key_i32("split.tensors.count", total);
    }
    // Pads the header to the 32-byte data section and appends the tensor data.
    Writer& data(const std::vector<uint8_t>& data) {
        bytes.resize((bytes.size() + 31) / 32 * 32);
        bytes.insert(bytes.end(), data.begin(), data.end());
        return *this;
    }
    void save(const std::filesystem::path& path) const {
        std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), std::ssize(bytes));
    }
};

// How split_model() lays out its shards, and the faults it can write.
struct SplitLayout {
    bool weights_in_shard1 = false;  // true: 2 shards, tensors in shard 1 too; false: 3 shards, shard 1 metadata only
    int32_t total = 3;               // every shard's split.tensors.count
    int last_index = -1;             // the last shard's split.no (-1: its right one)
    int count1 = 0;                  // shard 1's split.count (0: the right one)
    uint64_t a_offset = 0;           // tensor a's offset in its shard (a fault when not 0: past the data)
    uint64_t c_offset = 0;           // tensor c's offset in the last shard (the same)
    int shards() const { return weights_in_shard1 ? 2 : 3; }
};

// One F32 tensor of the split model: its name, its values (shard * 100,000 + tensor * 10,000 + i) and its shard.
struct SplitTensor {
    std::string name;
    int shard;
    std::vector<float> values;
};

// a (8 values) and b (16) in the first shard with weights, c (3,000: it crosses pages) in the last. b ends that
// shard's file inside a page.
inline std::vector<SplitTensor> split_tensors(const SplitLayout& layout = {}) {
    const int first = layout.weights_in_shard1 ? 1 : 2, last = layout.shards();
    std::vector<SplitTensor> out = {{"a", first, std::vector<float>(8)},
                                    {"b", first, std::vector<float>(16)},
                                    {"c", last, std::vector<float>(3000)}};
    for (size_t t = 0; t < out.size(); ++t)
        for (size_t i = 0; i < out[t].values.size(); ++i)
            out[t].values[i] =
                static_cast<float>(out[t].shard * 100000 + static_cast<int>(t) * 10000 + static_cast<int>(i));
    return out;
}

inline std::vector<uint8_t> bytes_of(const std::vector<float>& v) {
    const auto* p = reinterpret_cast<const uint8_t*>(v.data());
    return {p, p + v.size() * sizeof(float)};
}

// Writes DIR/NAME-0000k-of-0000N.gguf for the layout's N shards and returns shard 1's path.
inline std::filesystem::path split_model(const std::filesystem::path& dir, const std::string& name,
                                         const SplitLayout& layout = {}) {
    constexpr auto kF32 = static_cast<uint32_t>(Type::F32);
    const std::vector<SplitTensor> t = split_tensors(layout);
    const int n = layout.shards();
    const auto path = [&](int k) {
        char s[32];
        std::snprintf(s, sizeof s, "-%05d-of-%05d.gguf", k, n);
        return dir / (name + s);
    };
    const auto count = [&](int k) { return static_cast<uint16_t>(k == 1 && layout.count1 ? layout.count1 : n); };
    std::vector<uint8_t> ab = bytes_of(t[0].values), b = bytes_of(t[1].values);
    ab.insert(ab.end(), b.begin(), b.end());
    Writer first;
    first.header(layout.weights_in_shard1 ? 2 : 0, 4)
        .key_string("general.architecture", "qwen4exp")
        .split(0, count(1), layout.total);
    if (layout.weights_in_shard1)
        first.tensor("a", {8}, kF32, layout.a_offset).tensor("b", {16}, kF32, 32).data(ab);
    first.save(path(1));
    if (!layout.weights_in_shard1)
        Writer()
            .header(2, 3)
            .split(1, count(2), layout.total)
            .tensor("a", {8}, kF32, 0)
            .tensor("b", {16}, kF32, 32)
            .data(ab)
            .save(path(2));
    Writer()
        .header(1, 3)
        .split(static_cast<uint16_t>(layout.last_index >= 0 ? layout.last_index : n - 1), count(n), layout.total)
        .tensor("c", {3000}, kF32, layout.c_offset)
        .data(bytes_of(t[2].values))
        .save(path(n));
    return path(1);
}

}  // namespace torch::test
