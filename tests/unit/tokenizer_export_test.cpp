// Checks gguf::export_tokenizer on small headers built here: the files it writes (JSON escapes, the added tokens'
// special flags, unused ids left out, the template byte for byte) and the files it refuses. The real vocabulary is
// checked against the base model's tokenizer by tests/tools/tokenizer_export_test.py (CI job tokenizer-export).
#include "gguf/tokenizer_export.h"

#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "check.h"
#include "core/error.h"

using namespace torch;
using gguf::File;
using gguf::ValueType;

namespace {

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
    Writer& header(uint64_t n_keys) {
        bytes.insert(bytes.end(), {'G', 'G', 'U', 'F'});
        return put<uint32_t>(3).put<uint64_t>(0).put(n_keys);
    }
    Writer& key_string(std::string_view key, std::string_view value) {
        return string(key).put(ValueType::String).string(value);
    }
    Writer& key_strings(std::string_view key, std::initializer_list<std::string_view> values) {
        string(key).put(ValueType::Array).put(ValueType::String).put<uint64_t>(values.size());
        for (std::string_view v : values)
            string(v);
        return *this;
    }
    Writer& key_i32s(std::string_view key, std::initializer_list<int32_t> values) {
        string(key).put(ValueType::Array).put(ValueType::I32).put<uint64_t>(values.size());
        for (int32_t v : values)
            put(v);
        return *this;
    }
};

// A header with the tokenizer keys; `skip` leaves one out.
std::vector<uint8_t> tokenizer_header(std::string_view pre, std::initializer_list<std::string_view> tokens,
                                      std::initializer_list<int32_t> types, std::string_view skip = "") {
    Writer w;
    w.header(skip.empty() ? 6 : 5);
    if (skip != "tokenizer.ggml.model")
        w.key_string("tokenizer.ggml.model", "gpt2");
    if (skip != "tokenizer.ggml.pre")
        w.key_string("tokenizer.ggml.pre", pre);
    if (skip != "tokenizer.ggml.tokens")
        w.key_strings("tokenizer.ggml.tokens", tokens);
    if (skip != "tokenizer.ggml.token_type")
        w.key_i32s("tokenizer.ggml.token_type", types);
    if (skip != "tokenizer.ggml.merges")
        w.key_strings("tokenizer.ggml.merges", {"a b"});
    if (skip != "tokenizer.chat_template")
        w.key_string("tokenizer.chat_template", "{{ messages }}\n");
    return w.bytes;
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
}

bool refuses(const std::vector<uint8_t>& header, const std::filesystem::path& dir) {
    try {
        gguf::export_tokenizer(File::parse(header), dir);
        return false;
    } catch (const Error&) {
        return true;
    }
}

}  // namespace

int main() {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "torch-tokenizer-export-test";
    std::filesystem::remove_all(dir);

    // normal, normal with a quote, normal backslash, normal, control special, user-defined, control not special, unused
    const auto header =
        tokenizer_header("qwen35", {"a", "\"q\"", "\\", "ab", "<|im_start|>", "<think>", "<|fim_pad|>", "[PAD7]"},
                         {1, 1, 1, 1, 3, 4, 3, 5});
    const gguf::TokenizerCounts n = gguf::export_tokenizer(File::parse(header), dir);
    CHECK(n.vocab == 4 && n.added == 3 && n.unused == 1 && n.merges == 1 && n.template_bytes == 15);
    const std::string json = read_text(dir / "tokenizer.json");
    CHECK(json.find("\"\\\"q\\\"\": 1") != std::string::npos);  // "\"q\"": 1
    CHECK(json.find("\"\\\\\": 2") != std::string::npos);       // "\\": 2
    CHECK(json.find("\"ab\": 3\n") != std::string::npos);
    CHECK(json.find("{\"id\": 4, \"content\": \"<|im_start|>\", \"single_word\": false, \"lstrip\": false, \"rstrip\": "
                    "false, \"normalized\": false, \"special\": true}") != std::string::npos);
    CHECK(json.find("\"<think>\", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, "
                    "\"special\": false}") != std::string::npos);
    CHECK(json.find("\"<|fim_pad|>\", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": "
                    "false, \"special\": false}") != std::string::npos);
    CHECK(json.find("PAD7") == std::string::npos);
    CHECK(json.find("\"merges\": [\n      \"a b\"\n    ]") != std::string::npos);
    CHECK(json.find("\"normalizer\": {\"type\": \"NFC\"}") != std::string::npos);
    CHECK(read_text(dir / "chat_template.jinja") == "{{ messages }}\n");

    CHECK(refuses(tokenizer_header("custom", {"a"}, {1}), dir));                             // another pre-tokenizer
    CHECK(refuses(tokenizer_header("qwen35", {"a", "b"}, {1}), dir));                        // types do not match
    CHECK(refuses(tokenizer_header("qwen35", {"a", "<x>", "b"}, {1, 3, 1}), dir));           // normal after added
    CHECK(refuses(tokenizer_header("qwen35", {"a", "[PAD1]", "<x>"}, {1, 5, 3}), dir));      // added after unused
    CHECK(refuses(tokenizer_header("qwen35", {"a", "b", "a"}, {1, 1, 1}), dir));             // a repeated token
    CHECK(refuses(tokenizer_header("qwen35", {"a"}, {1}, "tokenizer.chat_template"), dir));  // no template
    CHECK(refuses(tokenizer_header("qwen35", {"a"}, {1}, "tokenizer.ggml.merges"), dir));    // no merges

    std::filesystem::remove_all(dir);
    return test::test_exit_code();
}
