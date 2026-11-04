// Tensor types as GGUF stores them: the GGUF tensor type ids, their block sizes, and the 13 the engine dequantizes.
#pragma once

#include <cstdint>
#include <string_view>

namespace torch {

// GGUF tensor type ids. Only the 13 formats of the model's two shards are named; type_info() knows every id the
// format assigns, so the reader accepts files that contain other types.
enum class Type : uint32_t {
    F32 = 0,
    F16 = 1,
    Q8_0 = 8,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    IQ3_XXS = 18,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    BF16 = 30,
    Q2_0 = 42,
};

// A type stores block_elements values in block_bytes bytes; rows are whole blocks.
struct TypeInfo {
    std::string_view name;
    uint32_t block_elements;
    uint32_t block_bytes;
};

// nullptr for an id the format does not assign.
const TypeInfo* type_info(uint32_t id) noexcept;

// Bytes of a row of `elements` values. elements % t.block_elements == 0.
constexpr uint64_t row_bytes(const TypeInfo& t, uint64_t elements) {
    return elements / t.block_elements * t.block_bytes;
}

// Calls f.template operator()<T>() with the run-time type as a template argument. Returns false, without calling f,
// for a type the engine does not dequantize.
template <class F>
bool with_type(Type type, F&& f) {
    switch (type) {
        case Type::F32:
            f.template operator()<Type::F32>();
            return true;
        case Type::F16:
            f.template operator()<Type::F16>();
            return true;
        case Type::Q8_0:
            f.template operator()<Type::Q8_0>();
            return true;
        case Type::Q4_K:
            f.template operator()<Type::Q4_K>();
            return true;
        case Type::Q5_K:
            f.template operator()<Type::Q5_K>();
            return true;
        case Type::Q6_K:
            f.template operator()<Type::Q6_K>();
            return true;
        case Type::IQ3_XXS:
            f.template operator()<Type::IQ3_XXS>();
            return true;
        case Type::IQ4_NL:
            f.template operator()<Type::IQ4_NL>();
            return true;
        case Type::IQ3_S:
            f.template operator()<Type::IQ3_S>();
            return true;
        case Type::IQ2_S:
            f.template operator()<Type::IQ2_S>();
            return true;
        case Type::IQ4_XS:
            f.template operator()<Type::IQ4_XS>();
            return true;
        case Type::BF16:
            f.template operator()<Type::BF16>();
            return true;
        case Type::Q2_0:
            f.template operator()<Type::Q2_0>();
            return true;
    }
    return false;
}

// True for the 13 named Types.
inline bool dequantizes(uint32_t id) noexcept {
    return with_type(static_cast<Type>(id), []<Type>() {});
}

}  // namespace torch
