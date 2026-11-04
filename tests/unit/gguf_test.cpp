// Checks the GGUF reader on tests/fixtures/tiny.gguf (written by gguf_writer.py) and on corrupt headers built here.
#include "gguf/gguf.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <string>
#include <vector>

#include "check.h"
#include "core/error.h"
#include "formats/types.h"

using namespace torch;
using gguf::File;
using gguf::ValueType;

namespace {

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
}

std::filesystem::path write_temp(const std::vector<uint8_t>& bytes, const char* name) {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), std::ssize(bytes));
    return path;
}

// True when parsing fails the way corrupt input must: with torch::Error. Any other exception escapes the test.
bool rejects(std::vector<uint8_t> bytes) {
    try {
        File::parse(std::move(bytes));
        return false;
    } catch (const Error&) {
        return true;
    }
}

bool opens(const std::filesystem::path& path, bool header_only) {
    try {
        File::open(path, header_only);
        return true;
    } catch (const Error&) {
        return false;
    }
}

template <class T>
void patch(std::vector<uint8_t>& bytes, size_t at, T value) {
    std::memcpy(bytes.data() + at, &value, sizeof value);
}

// Builds headers field by field, for the malformed files gguf_writer.py cannot write.
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
    Writer& key_u32(std::string_view key, uint32_t value) { return string(key).put(ValueType::U32).put(value); }
    Writer& tensor(std::string_view name, std::initializer_list<int64_t> shape, uint32_t type, uint64_t offset) {
        string(name).put(static_cast<uint32_t>(shape.size()));
        for (int64_t d : shape)
            put(d);
        return put(type).put(offset);
    }
};

std::vector<uint8_t> one_tensor(std::initializer_list<int64_t> shape, uint32_t type, uint64_t offset = 0) {
    return Writer().header(1, 0).tensor("t", shape, type, offset).bytes;
}

constexpr auto kF32 = static_cast<uint32_t>(Type::F32);
constexpr auto kQ2_0 = static_cast<uint32_t>(Type::Q2_0);

void check_tiny(const std::filesystem::path& path) {
    const File f = File::open(path);
    CHECK(f.version() == 3);
    CHECK(f.alignment() == 32);
    CHECK(f.data_offset() == 480);
    CHECK(f.file_size() == 640);
    CHECK(f.metadata().size() == 7);
    CHECK(f.metadata()[0].first == "general.architecture");

    CHECK(f.get<std::string>("general.architecture") == "qwen4exp");
    CHECK(f.get<uint32_t>("qwen4exp.block_count") == 48u);
    CHECK(f.get<int64_t>("qwen4exp.block_count") == 48);
    CHECK(f.get<uint16_t>("qwen4exp.block_count") == uint16_t{48});
    CHECK(f.get<float>("qwen4exp.attention.layer_norm_rms_epsilon") == 1e-6f);
    CHECK(f.get<std::vector<std::string>>("tokenizer.ggml.tokens") == std::vector<std::string>({"a", "bc", "def"}));
    CHECK(!f.get<uint32_t>("general.architecture"));  // wrong type
    CHECK(!f.get<std::string>("qwen4exp.block_count"));
    CHECK(!f.get<std::vector<uint32_t>>("tokenizer.ggml.tokens"));
    CHECK(!f.get<float>("qwen4exp.block_count"));
    CHECK(!f.get<uint32_t>("no.such.key"));
    CHECK(f.find("no.such.key") == nullptr);
    const gguf::Value* tokens = f.find("tokenizer.ggml.tokens");
    CHECK(tokens && tokens->type == ValueType::Array && tokens->element == ValueType::String && tokens->count == 3);

    const std::optional<gguf::Split> split = f.split();
    CHECK(split && split->index == 0 && split->count == 2 && split->total_tensors == 3);

    CHECK(f.tensors().size() == 2);
    CHECK(f.tensor("missing") == nullptr);
    const gguf::TensorInfo* norm = f.tensor("output_norm.weight");
    CHECK(norm && norm == &f.tensors()[0]);
    if (norm) {
        CHECK(norm->type == kF32 && norm->n_dims == 2 && norm->shape == (std::array<int64_t, 4>{3, 2, 1, 1}));
        CHECK(norm->offset == 0 && norm->nbytes == 24 && norm->elements() == 6 && norm->row_bytes() == 12);
        float values[6] = {};
        f.read(*norm, 0, sizeof values, values);
        CHECK(values[0] == 0.0f && values[5] == 5.0f);
    }

    const gguf::TensorInfo* experts = f.tensor("blk.0.ffn_down_exps.weight");
    CHECK(experts != nullptr);
    if (experts) {
        CHECK(experts->type == kQ2_0 && experts->n_dims == 3);
        CHECK(experts->shape == (std::array<int64_t, 4>{64, 2, 3, 1}));
        CHECK(experts->offset == 32 && experts->nbytes == 3 * 2 * 18 && experts->elements() == 384);
        CHECK(experts->row_bytes() == 18 && experts->slice_bytes() == 36 && experts->slice_offset(2) == 72);

        // read() of one expert equals the same bytes through the mapping.
        std::vector<uint8_t> slice(experts->slice_bytes());
        f.read(*experts, experts->slice_offset(2), slice.size(), slice.data());
        const std::span<const uint8_t> all = f.mapped(*experts);
        CHECK(all.size() == experts->nbytes);
        CHECK(std::equal(slice.begin(), slice.end(), all.begin() + 72));

        bool threw = false;
        try {
            f.read(*experts, experts->slice_offset(2), experts->slice_bytes() + 1, slice.data());
        } catch (const Error&) {
            threw = true;
        }
        CHECK(threw);  // past the end of the tensor
    }
}

// A file cut inside its tensor data opens only with header_only; read() and mapped() then refuse the missing bytes.
void check_header_only(const std::vector<uint8_t>& tiny) {
    const std::vector<uint8_t> cut(tiny.begin(), tiny.begin() + 600);
    const std::filesystem::path path = write_temp(cut, "torch-gguf-test-cut.gguf");
    CHECK(!opens(path, false));
    CHECK(opens(path, true));
    const File f = File::open(path, true);
    CHECK(f.tensors().size() == 2 && f.file_size() == 600);
    CHECK(f.mapped(*f.tensor("output_norm.weight")).size() == 24);
    bool read_threw = false, map_threw = false;
    std::vector<uint8_t> buffer(108);
    try {
        f.read(*f.tensor("blk.0.ffn_down_exps.weight"), 0, buffer.size(), buffer.data());
    } catch (const Error&) {
        read_threw = true;
    }
    try {
        f.mapped(*f.tensor("blk.0.ffn_down_exps.weight"));
    } catch (const Error&) {
        map_threw = true;
    }
    CHECK(read_threw && map_threw);
    std::filesystem::remove(path);
}

// Every prefix that stops inside the header is rejected; every longer prefix parses. None may crash.
void check_truncation(const std::vector<uint8_t>& tiny) {
    size_t first_ok = tiny.size() + 1;
    for (size_t cut = 0; cut <= tiny.size(); ++cut) {
        const bool rejected = rejects({tiny.begin(), tiny.begin() + static_cast<std::ptrdiff_t>(cut)});
        if (!rejected && first_ok > tiny.size())
            first_ok = cut;
        if (first_ok <= tiny.size())
            CHECK(!rejected);
    }
    CHECK(first_ok == 454);  // the end of the tensor table; the data section starts at the next 32-byte boundary
    CHECK(!opens(write_temp({}, "torch-gguf-test-empty.gguf"), true));
    CHECK(!opens("/nonexistent/torch.gguf", true));
}

void check_corruptions(const std::vector<uint8_t>& tiny) {
    std::vector<uint8_t> bad = tiny;
    bad[0] = 'X';
    CHECK(rejects(bad));  // magic

    bad = tiny;
    patch<uint32_t>(bad, 4, 2);
    CHECK(rejects(bad));  // version 2

    bad = tiny;
    patch<uint64_t>(bad, 8, UINT64_MAX);
    CHECK(rejects(bad));  // tensor count
    bad = tiny;
    patch<uint64_t>(bad, 16, uint64_t{1} << 40);
    CHECK(rejects(bad));  // key count
    bad = tiny;
    patch<uint64_t>(bad, 24, UINT64_MAX);
    CHECK(rejects(bad));  // length of the first key

    // Tensor tables. parse() does not check tensor bounds, so these headers need no data.
    CHECK(!rejects(one_tensor({64, 2}, kQ2_0)));
    CHECK(rejects(one_tensor({}, kF32)));                                    // 0 dimensions
    CHECK(rejects(one_tensor({1, 1, 1, 1, 1}, kF32)));                       // 5 dimensions
    CHECK(rejects(one_tensor({0}, kF32)));                                   // empty dimension
    CHECK(rejects(one_tensor({-8}, kF32)));                                  // negative dimension
    CHECK(rejects(one_tensor({int64_t{1} << 41}, kF32)));                    // dimension above 2^40
    CHECK(rejects(one_tensor({int64_t{1} << 40, int64_t{1} << 40}, kF32)));  // size overflows
    CHECK(rejects(one_tensor({8}, 1000)));                                   // unknown type
    CHECK(rejects(one_tensor({8}, 4)));                                      // an unassigned type id
    CHECK(rejects(one_tensor({32}, kQ2_0)));                                 // half a 64-element block
    CHECK(rejects(one_tensor({8}, kF32, 16)));                               // offset not 32-byte aligned
    CHECK(rejects(Writer().header(2, 0).tensor("t", {8}, kF32, 0).tensor("t", {8}, kF32, 32).bytes));  // same name

    // Metadata.
    CHECK(!rejects(Writer().header(0, 1).key_u32("general.alignment", 64).bytes));
    CHECK(File::parse(Writer().header(0, 1).key_u32("general.alignment", 64).bytes).alignment() == 64);
    CHECK(rejects(Writer().header(0, 1).key_u32("general.alignment", 0).bytes));
    CHECK(rejects(Writer().header(0, 1).key_u32("general.alignment", 48).bytes));
    CHECK(rejects(Writer().header(0, 2).key_u32("a", 1).key_u32("a", 2).bytes));                 // duplicate key
    CHECK(rejects(Writer().header(0, 1).string("a").put<uint32_t>(13).put<uint32_t>(0).bytes));  // unknown type
    CHECK(rejects(Writer().header(0, 1).string("a").put(ValueType::String).put(UINT64_MAX).bytes));
    const auto array = [](ValueType element, uint64_t count) {
        return Writer().header(0, 1).string("a").put(ValueType::Array).put(element).put(count).bytes;
    };
    CHECK(!rejects(array(ValueType::U32, 0)));
    CHECK(rejects(array(ValueType::Array, 0)));             // nested array
    CHECK(rejects(array(ValueType::U64, UINT64_MAX / 4)));  // more elements than bytes
    CHECK(rejects(array(ValueType::String, UINT64_MAX)));
    CHECK(rejects(Writer().header(0, 1).key_u32("split.count", 2).bytes));  // split.no missing
    CHECK(rejects(Writer()
                      .header(0, 3)
                      .key_u32("split.no", 2)
                      .key_u32("split.count", 2)
                      .key_u32("split.tensors.count", 5)
                      .bytes));  // index out of range
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: gguf_test tiny.gguf\n");
        return 2;
    }
    const std::vector<uint8_t> tiny = read_file(argv[1]);
    CHECK(tiny.size() == 640);
    if (tiny.size() != 640)
        return test::test_exit_code();

    check_tiny(argv[1]);
    check_header_only(tiny);
    check_truncation(tiny);
    check_corruptions(tiny);
    return test::test_exit_code();
}
