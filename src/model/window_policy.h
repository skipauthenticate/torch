// The adaptive width of a verify window: how many drafts to propose before verifying.
//
// A window of d drafts has d + 1 columns (the last accepted token and the drafts) and yields 1 + (accepted drafts)
// tokens. The policy picks the d that maximises expected tokens per millisecond:
//   q_i = a(p_1) * ... * a(p_i)   the chance that the first i drafts are all accepted, with p_j the drafter's
//                                 probability for draft j and a(p) the learned acceptance rate of drafts in p's bucket
//   E(d) = 1 + q_1 + ... + q_d    expected tokens of a window of d drafts
//   choose d maximising E(d) / cost(d), cost(d) the measured milliseconds of a window of d drafts, drafting included.
// a(p) is learned per probability bucket (8 buckets over [0, 1]) from every verified window: each draft the target
// accepted is a success, the draft it rejected is a failure, and later drafts were never tested so they teach nothing.
// The rates start from priors (about 30% below p = 0.5, 98% from p = 0.99 up) worth a few dozen observations.
// cost(d) starts at the seed values and follows an exponential moving average of the observed times.
//
// A policy that has stopped drafting would stop learning too, so after 32 windows without drafts it verifies one draft.
// The policy is deterministic and only decides how many drafts are verified, so greedy output never depends on it.
//
// Use: after each draft step call more(); when it returns false (or max_drafts is reached) read width(), verify that
// many drafts, then call observe().
//
// Test: tests/unit/window_policy_test.cpp.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace torch::model {

// Propose a continuation only when every complete earlier match of the last eight tokens agrees.
std::span<const int32_t> matching_continuation(std::span<const int32_t> history, int depth);

class WindowPolicy {
public:
    static constexpr int kBuckets = 8;

    // seed_costs[d] is the expected milliseconds of a window of d drafts, for d = 0 .. max_drafts (one more than
    // max_drafts entries, all positive); max_drafts is at most 7, the window having at most 8 columns. With `tight`,
    // more() also stops once width() would not verify every draft so far (a draft that is drafted and then not
    // verified costs a full draft pass for nothing).
    WindowPolicy(int max_drafts, std::vector<double> seed_costs, bool tight = false);

    // The drafter has produced draft number `drafted` (1 for the first of a window) with probability p_last. Returns
    // true when another draft is worth drafting: fewer than max_drafts so far and one or two more drafts, accepted as
    // often as the best bucket accepts, could still beat the best expected tokens per millisecond of this window.
    bool more(int drafted, float p_last);

    // The number of drafts to verify for the drafts of the current window: the d in 0 .. drafted that maximises
    // E(d) / cost(d); the smaller d on a tie. Remembers the probabilities of the drafts it chose, so observe() may be
    // called after the next window's drafting has begun.
    int width();

    // After a window verified `drafts` drafts and the target accepted `accepted` of them (0 .. drafts), which took
    // window_ms in all: updates the acceptance rates (from the current window's draft probabilities) and the cost.
    void observe(int drafts, int accepted, double window_ms);

    // The learned acceptance rate for a draft of probability p, and the current cost estimate of d drafts.
    double acceptance(float p) const;
    double cost(int drafts) const { return cost_[static_cast<size_t>(drafts)]; }
    int max_drafts() const { return max_drafts_; }
    bool tight() const { return tight_; }

private:
    static int bucket(float p);
    double expected_per_ms(int drafts, double next_acceptance) const;

    int max_drafts_;
    bool tight_;
    std::vector<double> cost_;
    std::array<double, kBuckets> accepted_{}, trials_{};  // observations including the prior
    std::vector<float> verified_;                         // probabilities of the drafts width() chose, for observe()
    std::vector<float> probabilities_;                    // of the current window's drafts, in order
    int idle_windows_ = 0;                                // consecutive windows that verified no draft
};

}  // namespace torch::model
