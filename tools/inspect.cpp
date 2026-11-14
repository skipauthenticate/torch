// `torch inspect`: what a GGUF file contains. Prints its split, metadata and bytes per tensor format and, on request,
// every tensor, as text or as JSON.
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "core/error.h"
#include "formats/types.h"
#include "gguf/gguf.h"
#include "gguf/tokenizer_export.h"

namespace torch {
namespace {

using gguf::ValueType;

constexpr const char* kUsage =
    "usage: torch inspect [--tensors] [--header-only] [--json] FILE...\n"
    "       torch inspect [--header-only] --export-tokenizer DIR FILE\n"
    "\n"
    "  --tensors           also list every tensor: name, format, shape, size and file offset\n"
    "  --header-only       accept a file that stops after its tensor table, such as the first 11 MiB of shard 1\n"
    "  --json              print one JSON object per file\n"
    "  --export-tokenizer  write DIR/tokenizer.json and DIR/chat_template.jinja from shard 1's header\n";

constexpr uint64_t kMaxArrayShown = 64;  // longer arrays (the tokenizer's) are shown by element type and length
constexpr uint64_t kTextArrayItems = 8;  // text output shows the first few elements of an array
constexpr size_t kTextStringBytes = 64;  // and the start of a long string

template <class T>
T load(const uint8_t* p) {
    T value;
    std::memcpy(&value, p, sizeof value);
    return value;
}

std::string with_commas(uint64_t n) {
    std::string s = std::to_string(n);
    for (size_t i = s.size(); i > 3; i -= 3)
        s.insert(i - 3, ",");
    return s;
}

// "1 tensor", "1,223 tensors".
std::string count(uint64_t n, const char* noun) {
    return with_commas(n) + " " + noun + (n == 1 ? "" : "s");
}

double gib(uint64_t bytes) {
    return static_cast<double>(bytes) / (1 << 30);
}
double mib(uint64_t bytes) {
    return static_cast<double>(bytes) / (1 << 20);
}

// A scalar as text, in a form JSON accepts: floats in their shortest exact form, non-finite floats as null.
std::string scalar_text(ValueType type, const uint8_t* p) {
    const auto shortest = [](auto x) -> std::string {
        if (!std::isfinite(x))
            return "null";
        char buffer[32];
        return {buffer, std::to_chars(buffer, buffer + sizeof buffer, x).ptr};
    };
    switch (type) {
        case ValueType::U8:
            return std::to_string(load<uint8_t>(p));
        case ValueType::I8:
            return std::to_string(load<int8_t>(p));
        case ValueType::U16:
            return std::to_string(load<uint16_t>(p));
        case ValueType::I16:
            return std::to_string(load<int16_t>(p));
        case ValueType::U32:
            return std::to_string(load<uint32_t>(p));
        case ValueType::I32:
            return std::to_string(load<int32_t>(p));
        case ValueType::U64:
            return std::to_string(load<uint64_t>(p));
        case ValueType::I64:
            return std::to_string(load<int64_t>(p));
        case ValueType::F32:
            return shortest(load<float>(p));
        case ValueType::F64:
            return shortest(load<double>(p));
        case ValueType::Bool:
            return *p != 0 ? "true" : "false";
        case ValueType::String:
        case ValueType::Array:
            break;
    }
    return "?";
}

std::string_view string_at(const uint8_t* p, uint64_t n) {
    return {reinterpret_cast<const char*>(p), n};
}

// The first `limit` elements of an array; strings as they are, numbers as scalar_text().
std::vector<std::string> array_items(const gguf::Value& v, uint64_t limit) {
    std::vector<std::string> items;
    const uint8_t* p = v.bytes;
    for (uint64_t i = 0; i < std::min(v.count, limit); ++i) {
        if (v.element == ValueType::String) {
            const uint64_t n = load<uint64_t>(p);
            items.emplace_back(string_at(p + 8, n));
            p += 8 + n;
        } else {
            items.push_back(scalar_text(v.element, p));
            p += gguf::scalar_size(v.element);
        }
    }
    return items;
}

std::string json_string(std::string_view s) {
    std::string out = "\"";
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
    return out + "\"";
}

// At most `n` bytes of s on one line, cut at a UTF-8 character boundary.
std::string clip(std::string_view s, size_t n) {
    std::string out(s);
    if (s.size() > n) {
        size_t cut = n;
        while (cut > 0 && (static_cast<unsigned char>(s[cut]) & 0xc0) == 0x80)
            --cut;  // inside a character
        out = std::string(s.substr(0, cut)) + "...";
    }
    std::replace_if(out.begin(), out.end(), [](char c) { return static_cast<unsigned char>(c) < 0x20; }, ' ');
    return out;
}

std::string type_text(const gguf::Value& v) {
    if (v.type != ValueType::Array)
        return std::string(gguf::value_type_name(v.type));
    return std::string(gguf::value_type_name(v.element)) + "[" + with_commas(v.count) + "]";
}

std::string value_text(const gguf::Value& v) {
    if (v.type == ValueType::String) {
        std::string text = clip(string_at(v.bytes, v.count), kTextStringBytes);
        if (v.count > kTextStringBytes)
            text += " (" + with_commas(v.count) + " bytes)";
        return text;
    }
    if (v.type != ValueType::Array)
        return scalar_text(v.type, v.bytes);
    if (v.count > kMaxArrayShown)
        return "";
    std::string text;
    for (const std::string& item : array_items(v, kTextArrayItems)) {
        if (!text.empty())
            text += ", ";
        text += v.element == ValueType::String ? "\"" + clip(item, 24) + "\"" : item;
    }
    return v.count > kTextArrayItems ? text + ", ..." : text;
}

std::string value_json(const gguf::Value& v) {
    if (v.type == ValueType::String)
        return json_string(string_at(v.bytes, v.count));
    if (v.type != ValueType::Array)
        return scalar_text(v.type, v.bytes);
    if (v.count > kMaxArrayShown) {
        return "{\"type\": \"" + std::string(gguf::value_type_name(v.element)) +
               "\", \"length\": " + std::to_string(v.count) + "}";
    }
    std::string json = "[";
    for (const std::string& item : array_items(v, v.count)) {
        if (json.size() > 1)
            json += ", ";
        json += v.element == ValueType::String ? json_string(item) : item;
    }
    return json + "]";
}

std::string shape_text(const gguf::TensorInfo& t, const char* separator) {
    std::string text;
    for (uint32_t d = 0; d < t.n_dims; ++d)
        text += (d > 0 ? separator : "") + std::to_string(t.shape[d]);
    return text;
}

struct FormatTotal {
    std::string_view name;
    uint64_t tensors;
    uint64_t bytes;
};

// Tensors and bytes per format, largest first.
std::vector<FormatTotal> format_totals(const gguf::File& f) {
    std::vector<FormatTotal> totals;
    for (const gguf::TensorInfo& t : f.tensors()) {
        const std::string_view name = type_info(t.type)->name;
        auto it = std::find_if(totals.begin(), totals.end(), [&](const FormatTotal& x) { return x.name == name; });
        if (it == totals.end())
            it = totals.insert(totals.end(), {name, 0, 0});
        it->tensors += 1;
        it->bytes += t.nbytes;
    }
    std::sort(totals.begin(), totals.end(), [](const FormatTotal& a, const FormatTotal& b) {
        return a.bytes != b.bytes ? a.bytes > b.bytes : a.name < b.name;
    });
    return totals;
}

void print_text(const char* path, const gguf::File& f, bool list_tensors) {
    std::printf("%s\n  GGUF version %u", path, f.version());
    if (const auto split = f.split()) {
        std::printf(", shard %u of %u (%s tensors in all shards)", split->index + 1u, unsigned{split->count},
                    with_commas(split->total_tensors).c_str());
    }
    std::printf(", %s, %s\n", count(f.metadata().size(), "metadata key").c_str(),
                count(f.tensors().size(), "tensor").c_str());
    std::printf("  alignment %llu, data section at byte %s, file size %s bytes\n",
                static_cast<unsigned long long>(f.alignment()), with_commas(f.data_offset()).c_str(),
                with_commas(f.file_size()).c_str());

    std::printf("\n  metadata\n");
    for (const auto& [key, value] : f.metadata()) {
        const std::string text = value_text(value);
        if (text.empty()) {  // an array too long to show
            std::printf("    %-48s %s\n", key.c_str(), type_text(value).c_str());
        } else {
            std::printf("    %-48s %-16s %s\n", key.c_str(), type_text(value).c_str(), text.c_str());
        }
    }

    std::printf("\n  formats\n    %-8s %8s %18s %9s\n", "format", "tensors", "bytes", "GiB");
    uint64_t tensors = 0, bytes = 0;
    for (const FormatTotal& total : format_totals(f)) {
        std::printf("    %-8.*s %8s %18s %9.3f\n", static_cast<int>(total.name.size()), total.name.data(),
                    with_commas(total.tensors).c_str(), with_commas(total.bytes).c_str(), gib(total.bytes));
        tensors += total.tensors;
        bytes += total.bytes;
    }
    std::printf("    %-8s %8s %18s %9.3f\n", "total", with_commas(tensors).c_str(), with_commas(bytes).c_str(),
                gib(bytes));

    if (!list_tensors)
        return;
    int width = 4;
    for (const gguf::TensorInfo& t : f.tensors())
        width = std::max(width, static_cast<int>(t.name.size()));
    std::printf("\n  tensors\n    %-*s %-8s %-24s %10s %14s\n", width, "name", "format", "shape", "MiB", "offset");
    for (const gguf::TensorInfo& t : f.tensors()) {
        const std::string_view format = type_info(t.type)->name;
        std::printf("    %-*s %-8.*s %-24s %10.2f %14llu\n", width, t.name.c_str(), static_cast<int>(format.size()),
                    format.data(), shape_text(t, " x ").c_str(), mib(t.nbytes),
                    static_cast<unsigned long long>(f.data_offset() + t.offset));
    }
}

void print_json(const char* path, const gguf::File& f, bool list_tensors) {
    std::string json = "{\n  \"file\": " + json_string(path) + ",\n  \"version\": " + std::to_string(f.version());
    if (const auto split = f.split()) {
        json += ",\n  \"split\": {\"index\": " + std::to_string(split->index) +
                ", \"count\": " + std::to_string(split->count) +
                ", \"total_tensors\": " + std::to_string(split->total_tensors) + "}";
    } else {
        json += ",\n  \"split\": null";
    }
    json += ",\n  \"alignment\": " + std::to_string(f.alignment());
    json += ",\n  \"data_offset\": " + std::to_string(f.data_offset());
    json += ",\n  \"file_size\": " + std::to_string(f.file_size());

    json += ",\n  \"metadata\": {";
    const char* separator = "\n    ";
    for (const auto& [key, value] : f.metadata()) {
        json += separator + json_string(key) + ": " + value_json(value);
        separator = ",\n    ";
    }
    json += "\n  },\n  \"formats\": [";
    separator = "\n    ";
    for (const FormatTotal& total : format_totals(f)) {
        json += separator + std::string("{\"format\": ") + json_string(total.name) +
                ", \"tensors\": " + std::to_string(total.tensors) + ", \"bytes\": " + std::to_string(total.bytes) + "}";
        separator = ",\n    ";
    }
    json += "\n  ]";

    if (list_tensors) {
        json += ",\n  \"tensors\": [";
        separator = "\n    ";
        for (const gguf::TensorInfo& t : f.tensors()) {
            json += separator + std::string("{\"name\": ") + json_string(t.name) +
                    ", \"format\": " + json_string(type_info(t.type)->name) + ", \"shape\": [" + shape_text(t, ", ") +
                    "], \"bytes\": " + std::to_string(t.nbytes) +
                    ", \"file_offset\": " + std::to_string(f.data_offset() + t.offset) + "}";
            separator = ",\n    ";
        }
        json += "\n  ]";
    }
    json += "\n}\n";
    std::fputs(json.c_str(), stdout);
}

}  // namespace

int inspect(int argc, char** argv) {
    bool list_tensors = false, header_only = false, json = false;
    const char* export_dir = nullptr;
    std::vector<const char*> paths;
    for (int i = 0; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--tensors") {
            list_tensors = true;
        } else if (arg == "--header-only") {
            header_only = true;
        } else if (arg == "--json") {
            json = true;
        } else if (arg == "--export-tokenizer" && i + 1 < argc) {
            export_dir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::fputs(kUsage, stdout);
            return 0;
        } else if (arg.starts_with("-")) {
            std::fprintf(stderr, "torch inspect: unknown option %s\n\n%s", argv[i], kUsage);
            return 2;
        } else {
            paths.push_back(argv[i]);
        }
    }
    if (paths.empty() || (export_dir != nullptr && (paths.size() != 1 || list_tensors || json))) {
        std::fputs(kUsage, stderr);
        return 2;
    }
    if (export_dir != nullptr) {
        try {
            const gguf::TokenizerCounts n = gguf::export_tokenizer(gguf::File::open(paths[0], header_only), export_dir);
            std::printf(
                "wrote %s/tokenizer.json (%s BPE tokens, %s added tokens, %s merges; %s unused ids left out) "
                "and %s/chat_template.jinja (%s bytes)\n",
                export_dir, with_commas(n.vocab).c_str(), with_commas(n.added).c_str(), with_commas(n.merges).c_str(),
                with_commas(n.unused).c_str(), export_dir, with_commas(n.template_bytes).c_str());
            return 0;
        } catch (const Error& e) {
            std::fprintf(stderr, "torch inspect: %s\n", e.what());
            return 1;
        }
    }

    for (size_t i = 0; i < paths.size(); ++i) {
        try {
            const gguf::File f = gguf::File::open(paths[i], header_only);
            if (i > 0 && !json)
                std::printf("\n");
            if (json) {
                print_json(paths[i], f, list_tensors);
            } else {
                print_text(paths[i], f, list_tensors);
            }
        } catch (const Error& e) {
            std::fflush(stdout);
            std::fprintf(stderr, "torch inspect: %s\n", e.what());
            return 1;
        }
    }
    return 0;
}

}  // namespace torch
