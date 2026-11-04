#include "formats/types.h"

namespace torch {
namespace {

struct Entry {
    uint32_t id;
    TypeInfo info;
};

// Every tensor type id the GGUF format assigns as of 2026-11-03, with block sizes.
// Ids 4, 5, 31-33 and 36-38 are unassigned, so files that use them are rejected.
constexpr Entry kTypes[] = {
    {0, {"F32", 1, 4}},         {1, {"F16", 1, 2}},         {2, {"Q4_0", 32, 18}},      {3, {"Q4_1", 32, 20}},
    {6, {"Q5_0", 32, 22}},      {7, {"Q5_1", 32, 24}},      {8, {"Q8_0", 32, 34}},      {9, {"Q8_1", 32, 36}},
    {10, {"Q2_K", 256, 84}},    {11, {"Q3_K", 256, 110}},   {12, {"Q4_K", 256, 144}},   {13, {"Q5_K", 256, 176}},
    {14, {"Q6_K", 256, 210}},   {15, {"Q8_K", 256, 292}},   {16, {"IQ2_XXS", 256, 66}}, {17, {"IQ2_XS", 256, 74}},
    {18, {"IQ3_XXS", 256, 98}}, {19, {"IQ1_S", 256, 50}},   {20, {"IQ4_NL", 32, 18}},   {21, {"IQ3_S", 256, 110}},
    {22, {"IQ2_S", 256, 82}},   {23, {"IQ4_XS", 256, 136}}, {24, {"I8", 1, 1}},         {25, {"I16", 1, 2}},
    {26, {"I32", 1, 4}},        {27, {"I64", 1, 8}},        {28, {"F64", 1, 8}},        {29, {"IQ1_M", 256, 56}},
    {30, {"BF16", 1, 2}},       {34, {"TQ1_0", 256, 54}},   {35, {"TQ2_0", 256, 66}},   {39, {"MXFP4", 32, 17}},
    {40, {"NVFP4", 64, 36}},    {41, {"Q1_0", 128, 18}},    {42, {"Q2_0", 64, 18}},     {43, {"F8_E4M3", 1, 1}},
};

}  // namespace

const TypeInfo* type_info(uint32_t id) noexcept {
    for (const Entry& e : kTypes) {
        if (e.id == id)
            return &e.info;
    }
    return nullptr;
}

}  // namespace torch
