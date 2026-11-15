// Checks expert placement: uniform k per layer, the capacity of a byte budget, the ranking by route counts (order,
// ties, skipping an expert too large for the space left), extending a placement once more memory is known, a
// limit on the count, and the route-count file.
//
// usage: placement_test DIR (a writable directory for the round-trip file)
#include "core/placement.h"

#include <filesystem>
#include <vector>

#include "check.h"
#include "core/error.h"

using namespace torch;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: placement_test DIR\n");
        return 2;
    }
    // Three layers of four experts; one expert of layer 0 takes 10 bytes, of layer 1 20, of layer 2 30.
    const std::vector<uint64_t> bytes = {10, 20, 30};

    const Placement u = place_uniform(2, 4, bytes);
    CHECK(u.count() == 6 && u.bytes == 2 * 60);
    CHECK(u.resident[1] == (std::vector<int32_t>{0, 1}));
    CHECK(uniform_capacity(119, 4, bytes) == 1);
    CHECK(uniform_capacity(120, 4, bytes) == 2);
    CHECK(uniform_capacity(10000, 4, bytes) == 4);

    RouteCounts c;
    c.n_layer = 3, c.n_expert = 4;
    c.counts = {5,   0, 9, 1,   // layer 0
                9,   3, 0, 0,   // layer 1: expert 0 ties with layer 0's expert 2
                100, 0, 0, 2};  // layer 2
    // Room 45: layer 2 expert 0 (30 bytes), then the tie at count 9, which goes to the lower layer: layer 0 expert 2
    // (10 bytes). Layer 1 expert 0 (20 bytes) no longer fits and is skipped, and so is everything after it.
    Placement r = place_ranked(c, bytes, 45);
    CHECK(r.bytes == 40);
    CHECK(r.resident[0] == (std::vector<int32_t>{2}));
    CHECK(r.resident[1].empty());
    CHECK(r.resident[2] == (std::vector<int32_t>{0}));
    // Room 70: 30 + 10 + 20 (layer 1 expert 0) + 10 (layer 0 expert 0) = 70.
    r = place_ranked(c, bytes, 70);
    CHECK(r.bytes == 70);
    CHECK(r.resident[0] == (std::vector<int32_t>{0, 2}));
    CHECK(r.resident[1] == (std::vector<int32_t>{0}));

    // Extending the room-45 placement by 30 bytes: the next ranked experts not yet resident, layer 1 expert 0 (20)
    // and layer 0 expert 0 (10); the two placements together are the room-75 ranking.
    const Placement more = extend_ranked(c, bytes, place_ranked(c, bytes, 45), 30);
    CHECK(more.bytes == 30);
    CHECK(more.resident[0] == (std::vector<int32_t>{0}));
    CHECK(more.resident[1] == (std::vector<int32_t>{0}));
    CHECK(more.resident[2].empty());
    // The uniform order: extending 2 per layer by 60 bytes adds expert 2 of every layer, as place_uniform(3) has.
    const Placement next = extend_ranked(uniform_ranking(3, 4), bytes, u, 60);
    CHECK(next.bytes == 60);
    for (const std::vector<int32_t>& layer : next.resident)
        CHECK(layer == (std::vector<int32_t>{2}));
    CHECK(place_ranked(uniform_ranking(3, 4), bytes, 120).resident == u.resident);
    // A limit stops the ranking at a count, in the same order: 2 experts of the room-70 four, then the other 2.
    const Placement two = place_ranked(c, bytes, 70, 2);
    CHECK(two.count() == 2 && two.bytes == 40);
    CHECK(two.resident[0] == (std::vector<int32_t>{2}) && two.resident[2] == (std::vector<int32_t>{0}));
    const Placement rest = extend_ranked(c, bytes, two, 1000, 2);
    CHECK(rest.count() == 2 && rest.bytes == 30);
    CHECK(rest.resident[0] == (std::vector<int32_t>{0}) && rest.resident[1] == (std::vector<int32_t>{0}));

    const std::filesystem::path path = std::filesystem::path(argv[1]) / "route_counts_test.txt";
    c.write(path);
    RouteCounts back = RouteCounts::read(path);
    CHECK(back.counts == c.counts && back.n_layer == 3 && back.n_expert == 4);
    back.add(c);
    CHECK(back.at(2, 0) == 200);
    bool threw = false;
    try {
        RouteCounts other;
        other.n_layer = 2, other.n_expert = 4, other.counts.assign(8, 1);
        back.add(other);
    } catch (const Error&) {
        threw = true;
    }
    CHECK(threw);
    std::filesystem::remove(path);
    return test::test_exit_code();
}
