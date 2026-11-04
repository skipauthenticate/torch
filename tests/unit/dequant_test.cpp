// Checks the CPU reference dequantization of every format against the numpy reference's values in
// tests/fixtures/quant_blocks.bin (written by tests/fixtures/make_fixtures.py).
#include "formats/dequant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <vector>

#include "check.h"

using namespace torch;

namespace {

// The half-precision cases normal floats never produce: zeros, subnormals, the largest finite value, infinities, NaN.
void check_fp16_edges() {
    CHECK(fp16_to_f32(0x0000) == 0.0f && !std::signbit(fp16_to_f32(0x0000)));
    CHECK(fp16_to_f32(0x8000) == 0.0f && std::signbit(fp16_to_f32(0x8000)));
    CHECK(fp16_to_f32(0x0001) == 0x1p-24f);
    CHECK(fp16_to_f32(0x83ff) == -1023 * 0x1p-24f);
    CHECK(fp16_to_f32(0x0400) == 0x1p-14f);
    CHECK(fp16_to_f32(0x3c00) == 1.0f);
    CHECK(fp16_to_f32(0x7bff) == 65504.0f);
    CHECK(fp16_to_f32(0xfc00) == -std::numeric_limits<float>::infinity());
    CHECK(std::isnan(fp16_to_f32(0x7e00)));
    CHECK(bf16_to_f32(0x3f80) == 1.0f);
    CHECK(bf16_to_f32(0xc2f7) == -123.5f);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: dequant_test quant_blocks.bin\n");
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    const std::vector<uint8_t> fixture{std::istreambuf_iterator<char>(in), {}};
    if (fixture.empty()) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }

    // Records: u32 type, u32 n, u32 nbytes, bytes[nbytes], f32[n].
    std::set<uint32_t> types;
    for (size_t pos = 0; pos < fixture.size();) {
        uint32_t head[3];
        if (fixture.size() - pos < sizeof head)
            break;
        std::memcpy(head, fixture.data() + pos, sizeof head);
        const auto [type, n, nbytes] = head;
        const uint8_t* raw = fixture.data() + pos + sizeof head;
        pos += sizeof head + nbytes + 4ull * n;
        if (pos > fixture.size())
            break;
        std::vector<float> want(n), got(n);
        std::memcpy(want.data(), raw + nbytes, 4ull * n);

        const TypeInfo* info = type_info(type);
        CHECK(info != nullptr && dequantizes(type));
        if (info == nullptr || !dequantizes(type))
            continue;
        CHECK(row_bytes(*info, n) == nbytes);
        CHECK(dequant_row(static_cast<Type>(type), raw, n, got.data()));

        float largest = 0, worst = 0;
        for (uint32_t i = 0; i < n; ++i) {
            largest = std::max(largest, std::fabs(want[i]));
            worst = std::max(worst, std::fabs(got[i] - want[i]));
        }
        const bool ok = worst <= 1e-6f * largest;
        CHECK(ok);
        std::printf("%-8s %u values, max |x| %.4g, max error %.3g %s\n", std::string(info->name).c_str(), n, largest,
                    worst, ok ? "ok" : "FAIL");
        types.insert(type);
    }
    // A format dropped from the fixture must fail loudly, not pass silently.
    CHECK(types.size() == 13);
    CHECK(!dequant_row(static_cast<Type>(2), nullptr, 0, nullptr));  // Q4_0 is in no shard

    check_fp16_edges();
    return test::test_exit_code();
}
