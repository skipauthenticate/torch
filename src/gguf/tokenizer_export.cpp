#include "gguf/tokenizer_export.h"

#include <array>
#include <cstdio>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "core/error.h"

namespace torch::gguf {
namespace {

// Token types as GGUF stores them.
constexpr int32_t kNormal = 1, kControl = 3, kUserDefined = 4, kUnused = 5;

// What `qwen35` stands for in the base model's tokenizer.json: the split before byte-level encoding, and the
// control tokens that are not special there (they stay in decoded text).
constexpr std::string_view kQwen35Split =
    R"re((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)re";
constexpr std::array<std::string_view, 6> kControlNotSpecial = {"<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
                                                                "<|fim_pad|>",    "<|repo_name|>",  "<|file_sep|>"};
constexpr std::string_view kByteLevel =
    R"({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false})";

void append_json_string(std::string& out, std::string_view s) {
    out += '"';
    for (const char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            char escape[8];
            std::snprintf(escape, sizeof escape, "\\u%04x", static_cast<unsigned>(c));
            out += escape;
        } else {
            out += c;
        }
    }
    out += '"';
}

template <class T>
T required(const File& f, std::string_view key) {
    std::optional<T> value = f.get<T>(key);
    if (!value)
        throw Error("The GGUF has no usable " + std::string(key) + ": export the tokenizer from shard 1 of the model.");
    return std::move(*value);
}

void write_file(const std::filesystem::path& path, std::string_view bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close();
    if (!out)
        throw Error("Could not write " + path.string() + ": check that the folder exists and is writable.");
}

}  // namespace

TokenizerCounts export_tokenizer(const File& f, const std::filesystem::path& dir) {
    const auto model = required<std::string>(f, "tokenizer.ggml.model");
    const auto pre = required<std::string>(f, "tokenizer.ggml.pre");
    if (model != "gpt2" || pre != "qwen35")
        throw Error("The tokenizer is " + model + " with pre-tokenizer " + pre +
                    "; only gpt2 with qwen35 can be exported: use the model files docs/model-files.md lists.");
    const auto tokens = required<std::vector<std::string>>(f, "tokenizer.ggml.tokens");
    const auto types = required<std::vector<int32_t>>(f, "tokenizer.ggml.token_type");
    const auto merges = required<std::vector<std::string>>(f, "tokenizer.ggml.merges");
    const auto chat_template = required<std::string>(f, "tokenizer.chat_template");
    if (types.size() != tokens.size())
        throw Error("The GGUF's token types do not match its tokens one for one: the file is damaged.");

    // normal ids, then added ones, then unused ones
    size_t vocab = 0;
    while (vocab < types.size() && types[vocab] == kNormal)
        ++vocab;
    size_t end = vocab;
    while (end < types.size() && (types[end] == kControl || types[end] == kUserDefined))
        ++end;
    for (size_t i = end; i < types.size(); ++i) {
        if (types[i] != kUnused)
            throw Error("Token " + std::to_string(i) + " has type " + std::to_string(types[i]) +
                        " out of the normal, added, unused order this export knows: the file is not a supported one.");
    }
    std::unordered_set<std::string_view> seen;
    for (size_t i = 0; i < vocab; ++i) {
        if (!seen.insert(tokens[i]).second)
            throw Error("Token " + std::to_string(i) + " repeats an earlier token's text: the file is damaged.");
    }

    std::string json;
    json.reserve(16 << 20);
    json += "{\n  \"version\": \"1.0\",\n  \"truncation\": null,\n  \"padding\": null,\n  \"added_tokens\": [";
    for (size_t i = vocab; i < end; ++i) {
        bool special = types[i] == kControl;
        for (const std::string_view name : kControlNotSpecial)
            special = special && tokens[i] != name;
        json += i == vocab ? "\n    {\"id\": " : ",\n    {\"id\": ";
        json += std::to_string(i) + ", \"content\": ";
        append_json_string(json, tokens[i]);
        json += ", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": ";
        json += special ? "true}" : "false}";
    }
    json +=
        "\n  ],\n  \"normalizer\": {\"type\": \"NFC\"},\n  \"pre_tokenizer\": {\"type\": \"Sequence\", "
        "\"pretokenizers\": [{\"type\": \"Split\", \"pattern\": {\"Regex\": ";
    append_json_string(json, kQwen35Split);
    json += "}, \"behavior\": \"Isolated\", \"invert\": false}, ";
    json += kByteLevel;
    json += "]},\n  \"post_processor\": ";
    json += kByteLevel;
    json += ",\n  \"decoder\": ";
    json += kByteLevel;
    json +=
        ",\n  \"model\": {\"type\": \"BPE\", \"dropout\": null, \"unk_token\": null, \"continuing_subword_prefix\": "
        "\"\", \"end_of_word_suffix\": \"\", \"fuse_unk\": false, \"byte_fallback\": false, \"ignore_merges\": "
        "false,\n    \"vocab\": {";
    for (size_t i = 0; i < vocab; ++i) {
        json += i == 0 ? "\n      " : ",\n      ";
        append_json_string(json, tokens[i]);
        json += ": " + std::to_string(i);
    }
    json += "\n    },\n    \"merges\": [";
    for (size_t i = 0; i < merges.size(); ++i) {
        json += i == 0 ? "\n      " : ",\n      ";
        append_json_string(json, merges[i]);
    }
    json += "\n    ]\n  }\n}\n";

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec)
        throw Error("Could not create " + dir.string() + ": " + ec.message() + ".");
    write_file(dir / "tokenizer.json", json);
    write_file(dir / "chat_template.jinja", chat_template);
    return {vocab, end - vocab, tokens.size() - end, merges.size(), chat_template.size()};
}

}  // namespace torch::gguf
