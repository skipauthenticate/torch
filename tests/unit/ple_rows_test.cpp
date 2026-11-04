// Checks the PLE n-gram hash: the row each head reads, and how missing predecessors and end-of-sequence tokens
// shape the window.
#include "runtime/ple_rows.h"

#include <vector>

#include "check.h"

using namespace torch;
using runtime::PleRows;

namespace {

constexpr int32_t kEos = 99;

model::Config small_config() {
    model::Config c;
    c.ple_ngram = 3;
    c.ple_heads_per_ngram = 2;
    c.ple_eos = kEos;
    c.ple_multipliers = {1, 10, 100};
    c.ple_head_vocab = {1000, 997, 1000, 991};
    c.ple_head_offsets = {0, 1000, 2000, 3000};
    return c;
}

std::vector<uint64_t> rows(const std::vector<int32_t>& history, size_t i) {
    return PleRows::row_ids(small_config(), history, i);
}

}  // namespace

int main() {
    // Token 5 after 3 after 2: bigram 5 ^ 3*10 = 27; trigram 27 ^ 2*100 = 211.
    const std::vector<uint64_t> r = rows({2, 3, 5}, 2);
    CHECK(r.size() == 4);
    CHECK(r[0] == 27 % 1000 + 0 && r[1] == 27 % 997 + 1000);
    CHECK(r[2] == 211 % 1000 + 2000 && r[3] == 211 % 991 + 3000);

    // Before the start of the sequence, predecessors read as EOS; an EOS predecessor cuts the window there.
    CHECK(rows({5}, 0) == rows({kEos, 5}, 1));
    CHECK(rows({5}, 0) == rows({7, kEos, 5}, 2));
    CHECK(rows({3, 5}, 1) == rows({kEos, 3, 5}, 2));
    CHECK(rows({3, 5}, 1) != rows({4, 3, 5}, 2));
    // The token's own EOS does not cut its context.
    CHECK(rows({3, kEos}, 1) != rows({kEos}, 0));
    return test::test_exit_code();
}
