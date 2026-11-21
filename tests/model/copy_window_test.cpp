#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/model.h"

using torch::model::Model;

namespace {
void expect(bool ok, const std::string& message) {
    if (!ok)
        throw std::runtime_error(message);
}
}  // namespace

int main(int argc, char** argv) try {
    if (argc != 2) {
        std::fprintf(stderr, "usage: copy_window_test MODEL.gguf\n");
        return 2;
    }
    torch::model::Options options;
    options.model = argv[1];
    options.context = options.prefill_columns = 4096;
    options.logits = false;
    options.resident = 128;
    Model model(options);
    constexpr int prefix = 2050, width = 128;
    expect(model.copy_window_limit() >= width, "The test window must fit in the prompt workspace.");
    std::vector<int32_t> prompt(prefix);
    for (int i = 0; i < prefix; ++i)
        prompt[static_cast<size_t>(i)] = 100 + i % 61;
    expect(model.prefill(prompt), model.error());
    std::vector<int32_t> gold{model.argmax(0)};
    for (int i = 0; i < width + 16; ++i) {
        expect(model.verify(std::span(gold).last(1)), model.error());
        gold.push_back(model.argmax(0));
    }
    std::array<int32_t, width> predictions{};
    for (const int bad : {-1, 1, 79, width - 1}) {
        model.rewind();
        expect(model.verify(std::span(gold).first(7)), model.error());
        model.lend_chunk();
        std::vector<int32_t> proposal(gold.begin() + 7, gold.begin() + 7 + width);
        if (bad >= 0)
            proposal[static_cast<size_t>(bad)] = (proposal[static_cast<size_t>(bad)] + 1) % 10000;
        expect(model.prefill(proposal, predictions), model.error());
        expect(model.prefill_end() == prefix, "Checking copied tokens replaced the prompt checkpoint.");
        const int accepted = bad < 0 ? width : bad;
        expect(std::equal(predictions.begin(), predictions.begin() + accepted, gold.begin() + 8),
               "A target prediction differs from single-token generation.");
        const int fed = 7 + (bad < 0 ? width : 0);
        expect(model.position() == prefix + fed, "The model kept tokens after a wrong proposal.");
        expect(model.verify(std::span(gold).subspan(static_cast<size_t>(fed), 8)), model.error());
        for (int t = 0; t < 8; ++t)
            expect(model.argmax(t) == gold[static_cast<size_t>(fed + t + 1)], "The next window used incorrect state.");
    }
    std::puts("The accepted copy and three rejected copies preserved the prompt checkpoint and next-token IDs.");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
