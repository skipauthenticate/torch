#include "model/window_policy.h"

#include <algorithm>
#include <cmath>

#include "core/error.h"

namespace torch::model {
namespace {

// Acceptance rate per probability bucket before any observation, and the weight of that prior in observations.
constexpr std::array<double, WindowPolicy::kBuckets> kPrior = {0.30, 0.30, 0.30, 0.30, 0.50, 0.65, 0.80, 0.93};
constexpr double kPriorWeight = 20.0;
constexpr double kMaxTrials = 2000.0;  // beyond this the counts are halved, so old behaviour fades
constexpr double kCostSmoothing = 0.2;
constexpr int kProbeEvery = 32;  // windows of no drafts after which one draft is verified anyway, to keep learning

}  // namespace

WindowPolicy::WindowPolicy(int max_drafts, std::vector<double> seed_costs, bool tight)
    : max_drafts_(max_drafts), tight_(tight), cost_(std::move(seed_costs)) {
    if (max_drafts < 0 || max_drafts > 7)
        throw Error("A window has at most 7 drafts; lower max_drafts.");
    if (cost_.size() != static_cast<size_t>(max_drafts) + 1)
        throw Error("Give one seed cost per window width, from 0 to max_drafts drafts.");
    for (double c : cost_)
        if (!(c > 0.0))
            throw Error("Seed costs must be positive milliseconds.");
    for (int b = 0; b < kBuckets; ++b) {
        trials_[static_cast<size_t>(b)] = kPriorWeight;
        accepted_[static_cast<size_t>(b)] = kPrior[static_cast<size_t>(b)] * kPriorWeight;
    }
}

int WindowPolicy::bucket(float p) {
    if (!(p > 0.0f))
        return 0;  // also NaN
    return std::min(kBuckets - 1, static_cast<int>(p * kBuckets));
}

double WindowPolicy::acceptance(float p) const {
    const size_t b = static_cast<size_t>(bucket(p));
    return accepted_[b] / trials_[b];
}

// E(d) / cost(d) for the current window's drafts; when next_acceptance > 0 the drafts past the recorded ones are
// assumed accepted with that probability each.
double WindowPolicy::expected_per_ms(int drafts, double next_acceptance) const {
    double chance = 1.0, expected = 1.0;
    for (int i = 0; i < drafts; ++i) {
        chance *= static_cast<size_t>(i) < probabilities_.size() ? acceptance(probabilities_[static_cast<size_t>(i)])
                                                                 : next_acceptance;
        expected += chance;
    }
    return expected / cost_[static_cast<size_t>(drafts)];
}

bool WindowPolicy::more(int drafted, float p_last) {
    if (drafted <= 1)
        probabilities_.clear();
    probabilities_.resize(static_cast<size_t>(drafted) - 1);  // a repeated call replaces the draft
    probabilities_.push_back(p_last);
    if (drafted >= max_drafts_)
        return false;
    double best = 0.0, optimistic = 0.0;
    int best_d = 0;
    for (int d = 0; d <= drafted; ++d)
        if (const double v = expected_per_ms(d, 0.0); v > best)
            best = v, best_d = d;
    if (tight_ && best_d < drafted)
        return false;
    double top = 0.0;
    for (size_t b = 0; b < kBuckets; ++b)
        top = std::max(top, accepted_[b] / trials_[b]);
    for (int extra = 1; extra <= 2 && drafted + extra <= max_drafts_; ++extra)
        optimistic = std::max(optimistic, expected_per_ms(drafted + extra, top));
    return optimistic > best;
}

int WindowPolicy::width() {
    int best_d = 0;
    double best = 0.0;
    const int most = std::min(static_cast<int>(probabilities_.size()), max_drafts_);
    for (int d = 0; d <= most; ++d) {
        const double value = expected_per_ms(d, 0.0);
        if (value > best) {
            best = value;
            best_d = d;
        }
    }
    // With no drafts verified nothing is learned, so a rate that fell too low would never recover.
    if (best_d == 0 && most >= 1 && idle_windows_ >= kProbeEvery)
        best_d = 1;
    verified_.assign(probabilities_.begin(), probabilities_.begin() + best_d);
    return best_d;
}

void WindowPolicy::observe(int drafts, int accepted, double window_ms) {
    drafts = std::clamp(drafts, 0, max_drafts_);
    idle_windows_ = drafts == 0 ? idle_windows_ + 1 : 0;
    accepted = std::clamp(accepted, 0, drafts);
    // Draft j (0-based) was accepted for j < accepted, rejected for j == accepted, and untested after.
    const int tested = std::min(drafts, accepted + 1);
    for (int j = 0; j < tested && static_cast<size_t>(j) < verified_.size(); ++j) {
        const size_t b = static_cast<size_t>(bucket(verified_[static_cast<size_t>(j)]));
        trials_[b] += 1.0;
        accepted_[b] += j < accepted ? 1.0 : 0.0;
        if (trials_[b] > kMaxTrials) {
            trials_[b] *= 0.5;
            accepted_[b] *= 0.5;
        }
    }
    if (window_ms > 0.0 && std::isfinite(window_ms)) {
        double& c = cost_[static_cast<size_t>(drafts)];
        c += kCostSmoothing * (window_ms - c);
    }
    verified_.clear();
}

}  // namespace torch::model
