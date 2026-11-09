// Checks the window policy with synthetic acceptance and costs: wide windows for confident drafts, narrow ones for
// weak drafts or steeply rising costs, learning of the acceptance rates and the costs, and determinism.
#include "model/window_policy.h"

#include <cmath>
#include <vector>

#include "check.h"
#include "core/error.h"

using namespace torch;
using model::WindowPolicy;

namespace {

bool throws(auto&& f) {
    try {
        f();
        return false;
    } catch (const Error&) {
        return true;
    }
}

// Seed costs of a window of d drafts: base + step * d milliseconds.
std::vector<double> linear_costs(int max_drafts, double base, double step) {
    std::vector<double> c;
    for (int d = 0; d <= max_drafts; ++d)
        c.push_back(base + step * d);
    return c;
}

// Drafts every draft with probability p while the policy asks for more, and returns the width it chooses.
int width_for(WindowPolicy& policy, float p, int max_drafts) {
    int drafted = 0;
    while (drafted < max_drafts) {
        ++drafted;
        if (!policy.more(drafted, p))
            break;
    }
    return policy.width();
}

void check_validation() {
    CHECK(throws([] { WindowPolicy(8, linear_costs(8, 30, 5)); }));  // more than 7 drafts
    CHECK(throws([] { WindowPolicy(3, linear_costs(2, 30, 5)); }));  // too few seed costs
    CHECK(throws([] { WindowPolicy(3, {30, 35, 0, 45}); }));         // a cost that is not positive
    CHECK(!throws([] { WindowPolicy(0, {30}); }));
    WindowPolicy none(0, {30});
    CHECK(!none.more(1, 0.99f) && none.width() == 0);
}

void check_shapes() {
    // confident drafts at a modest cost per column: the widest window
    WindowPolicy confident(7, linear_costs(7, 37, 7.5));
    CHECK(width_for(confident, 0.995f, 7) == 7);
    // weak drafts: at most one
    WindowPolicy weak(7, linear_costs(7, 37, 7.5));
    CHECK(width_for(weak, 0.1f, 7) <= 1);
    // steeply rising costs: no drafts even when they are confident
    std::vector<double> steep;
    for (int d = 0; d <= 7; ++d)
        steep.push_back(10 * std::pow(3.0, d));
    WindowPolicy expensive(7, steep);
    CHECK(width_for(expensive, 0.995f, 7) == 0);
    // free columns: width follows acceptance only, so even mediocre drafts go to the maximum
    WindowPolicy cheap(7, linear_costs(7, 37, 0.01));
    CHECK(width_for(cheap, 0.5f, 7) == 7);
    // a confident run followed by a weak draft: the window stops at the weak one
    WindowPolicy mixed(7, linear_costs(7, 37, 7.5));
    int drafted = 0;
    bool keep_going = true;
    for (float p : {0.995f, 0.995f, 0.995f, 0.05f}) {
        ++drafted;
        keep_going = mixed.more(drafted, p);
    }
    CHECK(mixed.width() == 3);
    (void)keep_going;
}

void check_learning() {
    // drafts of probability 0.95 that the target always rejects: the rate falls, windows narrow to nothing
    WindowPolicy rejected(7, linear_costs(7, 37, 7.5));
    const double before = rejected.acceptance(0.95f);
    int first_width = -1;
    for (int w = 0; w < 300; ++w) {
        const int width = width_for(rejected, 0.95f, 7);
        if (first_width < 0)
            first_width = width;
        rejected.observe(width, 0, 37 + 7.5 * width);
    }
    CHECK(first_width >= 6);
    CHECK(before > 0.9);
    CHECK(rejected.acceptance(0.95f) < 0.25);
    CHECK(width_for(rejected, 0.95f, 7) <= 1);  // 0, or the one draft verified now and then so it keeps learning
    // drafts of probability 0.2 that the target always accepts: the rate rises and windows widen
    WindowPolicy accepted(7, linear_costs(7, 37, 7.5));
    const int narrow = width_for(accepted, 0.2f, 7);
    for (int w = 0; w < 300; ++w) {
        const int width = width_for(accepted, 0.2f, 7);
        accepted.observe(width, width, 37 + 7.5 * width);
    }
    CHECK(accepted.acceptance(0.2f) > 0.9);
    CHECK(width_for(accepted, 0.2f, 7) > narrow);
    // only the tested drafts teach: a window that rejected its first draft says nothing about the second
    WindowPolicy tested(7, linear_costs(7, 37, 7.5));
    width_for(tested, 0.99f, 7);
    const double second = tested.acceptance(0.99f);
    tested.observe(7, 0, 90);  // one rejected draft, six untested ones in the same bucket
    const double after = tested.acceptance(0.99f);
    CHECK(after < second);
    CHECK(std::fabs(after - second * 20.0 / 21.0) < 1e-12);  // exactly one more trial, no success
}

void check_next_window_overlap() {
    // observe() after the next window's drafting began still learns from the window it describes
    WindowPolicy policy(7, linear_costs(7, 37, 7.5));
    policy.more(1, 0.99f);
    policy.more(2, 0.99f);
    const int width = policy.width();
    CHECK(width >= 1);
    policy.more(1, 0.05f);  // the next window's first draft
    const double low_before = policy.acceptance(0.05f), high_before = policy.acceptance(0.99f);
    policy.observe(width, 0, 80.0);  // rejected: a failure for the 0.99 bucket, nothing for the 0.05 bucket
    CHECK(policy.acceptance(0.05f) == low_before);
    CHECK(policy.acceptance(0.99f) < high_before);
}

void check_costs() {
    WindowPolicy policy(7, linear_costs(7, 37, 7.5));
    const double seed = policy.cost(3);
    for (int i = 0; i < 100; ++i) {
        width_for(policy, 0.9f, 7);
        policy.observe(3, 3, 100.0);
    }
    CHECK(std::fabs(policy.cost(3) - 100.0) < 1e-6);
    CHECK(seed < 100.0 && policy.cost(2) == 37 + 7.5 * 2);  // other widths keep their estimates
    policy.observe(3, 3, -5.0);                             // a nonsense time changes nothing
    CHECK(std::fabs(policy.cost(3) - 100.0) < 1e-6);
}

void check_deterministic() {
    WindowPolicy a(5, linear_costs(5, 40, 6)), b(5, linear_costs(5, 40, 6));
    bool same = true;
    for (int w = 0; w < 200; ++w) {
        const float p = 0.05f + 0.9f * static_cast<float>((w * 37) % 100) / 100.0f;
        const int wa = width_for(a, p, 5), wb = width_for(b, p, 5);
        same = same && wa == wb;
        const int accepted = (w * 7) % (wa + 1);
        a.observe(wa, accepted, 40 + 6.0 * wa + (w % 5));
        b.observe(wb, accepted, 40 + 6.0 * wb + (w % 5));
    }
    CHECK(same);
}

}  // namespace

int main() {
    check_validation();
    check_shapes();
    check_learning();
    check_next_window_overlap();
    check_costs();
    check_deterministic();
    return test::test_exit_code();
}
