// Branch-and-price kernel for one fixed-fleet Stage-2 assignment surrogate.
//
// Model (identical to customized-subprob/s2forward/subset_dp.py):
//
//     min  sum_j c_out_j s_j + sum_v theta_v
//     s.t. every active customer is served by at most one vehicle (else
//          outsourced), load(S_v) <= Q_v, y_v = [S_v non-empty],
//          theta_v = max(0, max_c beta_c + piY_c y_v + sum_{j in S_v} piAlpha_c[j]),
//          assignment order: score(S_{v_i}) >= score(S_{v_{i+1}}) inside a type.
//
// Decomposition.  A column is one vehicle together with the exact customer
// set it serves; its cost is the exact theta of that set.  The restricted
// master is then a pure set-partitioning LP with one convexity row per
// vehicle and one linear ordering row per consecutive same-type pair (the
// score is additive, so ordering rows stay linear in the columns).  Folding
// the max over cuts into the column cost is what makes the LP relaxation
// strong: the compact model only sees the convex envelope of the cut maxima at
// fractional alpha, which is far below the integer value once several cuts
// compete.
//
// Pricing is a min-max knapsack per vehicle (minimise over feasible sets the
// maximum over cuts of an affine function minus the dual prices).  It is
// solved exactly by depth-first branch-and-bound; each node is bounded by the
// fractional knapsack of a few "active" cuts and of their best pairwise mix
// (valid because max_r f_r >= w f_r1 + (1-w) f_r2 for any w in [0,1]).
// Restricting the bounding cuts keeps every bound valid (fewer cuts => smaller
// maximum) while leaf values always use the full cut set.
//
// Certification.  Every node bound is a Lagrangian bound
//     b.y + sum_v min_S rc_v(S) + sum_j min(0, rc(z_j)),
// which is valid for any dual vector with non-negative ordering duals, so
// neither the simplex nor the pricing search needs to be trusted beyond the
// exact pricing minimum (or its valid lower bound after a node limit).  The
// returned interval [lb, ub] carries an explicit floating-point margin; the
// Python caller re-costs the incumbent exactly.
//
// Build: cpp/build_bp.sh (pybind11 module s2_bp_kernel).

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace s2bp {

using i64 = long long;
using u64 = unsigned long long;
static constexpr double kInf = std::numeric_limits<double>::infinity();
static constexpr int kMaxCustomers = 64;

// --------------------------------------------------------------------------
// Problem data
// --------------------------------------------------------------------------

struct Cut {
    double a = 0.0;          // beta + piY: value at y = 1, alpha = 0
    std::vector<double> p;   // piAlpha over active customers
};

struct Vehicle {
    i64 cap = 0;
    int type = 0;
    double empty_theta = 0.0;    // max(0, max_c beta_c)
    std::vector<Cut> cuts;
};

struct Problem {
    int n = 0;
    int m = 0;
    std::vector<double> c_out;
    std::vector<i64> q;
    std::vector<i64> w;                      // exact additive score weights
    std::vector<double> w_lp;                // w / sum(w): ordering-row coefficients in [0, 1]
    std::vector<Vehicle> veh;
    std::vector<std::vector<int>> groups;    // same-type runs, canonical order
    std::vector<std::pair<int, int>> pairs;  // (first, second) ordering pairs
    // pair index where v is first / second (-1 when none)
    std::vector<int> pair_first_of;
    std::vector<int> pair_second_of;

    void finalize() {
        // Ordering rows compare exact integer scores; the LP only needs the
        // direction, so scale them to O(1) for a well-conditioned basis.
        double wsum = 0.0;
        for (i64 x : w) wsum += (double)x;
        w_lp.assign(n, 0.0);
        for (int j = 0; j < n; ++j) w_lp[j] = wsum > 0.0 ? (double)w[j] / wsum : 0.0;
        groups.clear();
        pairs.clear();
        pair_first_of.assign(m, -1);
        pair_second_of.assign(m, -1);
        for (int v = 0; v < m; ++v) {
            if (v > 0 && veh[v].type == veh[v - 1].type) {
                groups.back().push_back(v);
                pair_first_of[v - 1] = (int)pairs.size();
                pair_second_of[v] = (int)pairs.size();
                pairs.emplace_back(v - 1, v);
            } else {
                groups.push_back({v});
            }
        }
        for (Vehicle& vh : veh) drop_dominated_cuts(vh);
    }

    // Remove cuts that are pointwise dominated on the active customers.  The
    // maximum over the remaining cuts is unchanged for every set.
    static void drop_dominated_cuts(Vehicle& vh) {
        const int K = (int)vh.cuts.size();
        std::vector<char> dead(K, 0);
        for (int r = 0; r < K; ++r) {
            if (dead[r]) continue;
            for (int s = 0; s < K; ++s) {
                if (s == r || dead[s]) continue;
                const Cut& cr = vh.cuts[r];
                const Cut& cs = vh.cuts[s];
                if (cs.a < cr.a) continue;
                bool dom = true;
                for (size_t j = 0; j < cr.p.size(); ++j) {
                    if (cs.p[j] < cr.p[j]) { dom = false; break; }
                }
                if (!dom) continue;
                // r <= s pointwise. Keep s (drop r); on exact ties drop the
                // later index so that at least one survives.
                if (cs.a == cr.a && cs.p == cr.p && s < r) continue;
                dead[r] = 1;
                break;
            }
        }
        std::vector<Cut> kept;
        for (int r = 0; r < K; ++r) if (!dead[r]) kept.push_back(std::move(vh.cuts[r]));
        vh.cuts = std::move(kept);
    }
};

inline double theta_mask(const Vehicle& vh, u64 mask) {
    if (mask == 0) return vh.empty_theta;
    double best = 0.0;
    for (const Cut& c : vh.cuts) {
        double s = c.a;
        for (u64 mm = mask; mm; mm &= mm - 1) s += c.p[__builtin_ctzll(mm)];
        if (s > best) best = s;
    }
    return best;
}

inline i64 mask_sum(const std::vector<i64>& vals, u64 mask) {
    i64 s = 0;
    for (u64 mm = mask; mm; mm &= mm - 1) s += vals[__builtin_ctzll(mm)];
    return s;
}

struct Clock {
    std::chrono::steady_clock::time_point start;
    double limit;
    Clock(double limit_seconds)
        : start(std::chrono::steady_clock::now()), limit(limit_seconds) {}
    double elapsed() const {
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - start).count();
    }
    bool expired() const { return elapsed() >= limit; }
};

// --------------------------------------------------------------------------
// Dense revised primal simplex (restricted master; <= ~70 rows)
// --------------------------------------------------------------------------

class DenseSimplex {
public:
    struct Col {
        std::vector<std::pair<int, double>> a;
        double c = 0.0;
        bool active = true;
    };
    enum Status { OPTIMAL = 0, ITER_LIMIT = 1, UNBOUNDED = 2, SINGULAR = 3 };

    DenseSimplex(int rows, std::vector<double> rhs)
        : R_(rows), b_(std::move(rhs)) {}

    int add(Col col) {
        cols_.push_back(std::move(col));
        pos_.push_back(-1);
        return (int)cols_.size() - 1;
    }
    int rows() const { return R_; }
    int num_cols() const { return (int)cols_.size(); }
    const Col& col(int k) const { return cols_[k]; }
    void set_active(int k, bool on) { cols_[k].active = on; }

    bool set_basis(const std::vector<int>& basic) {
        basis_ = basic;
        std::fill(pos_.begin(), pos_.end(), -1);
        for (int r = 0; r < R_; ++r) pos_[basis_[r]] = r;
        return refactor();
    }

    double objective() const {
        double z = 0.0;
        for (int r = 0; r < R_; ++r) z += cols_[basis_[r]].c * xB_[r];
        return z;
    }
    const std::vector<double>& duals() const { return y_; }
    double primal(int k) const {
        int r = pos_[k];
        return r < 0 ? 0.0 : xB_[r];
    }
    bool is_basic(int k) const { return pos_[k] >= 0; }
    double reduced_cost(int k) const {
        const Col& c = cols_[k];
        double d = c.c;
        for (const auto& e : c.a) d -= y_[e.first] * e.second;
        return d;
    }

    Status solve(long max_iters, std::mt19937& rng);

private:
    bool refactor();
    void compute_duals() {
        std::fill(y_.begin(), y_.end(), 0.0);
        for (int r = 0; r < R_; ++r) {
            double cb = cols_[basis_[r]].c;
            if (cb == 0.0) continue;
            const double* row = &Binv_[(size_t)r * R_];
            for (int i = 0; i < R_; ++i) y_[i] += cb * row[i];
        }
    }

    int R_;
    std::vector<double> b_;
    std::vector<Col> cols_;
    std::vector<int> pos_;
    std::vector<int> basis_;
    std::vector<double> Binv_, xB_, y_;
    int since_refactor_ = 0;
};

bool DenseSimplex::refactor() {
    const int R = R_;
    std::vector<double> B((size_t)R * R, 0.0), inv((size_t)R * R, 0.0);
    for (int r = 0; r < R; ++r) {
        for (const auto& e : cols_[basis_[r]].a) B[(size_t)e.first * R + r] += e.second;
        inv[(size_t)r * R + r] = 1.0;
    }
    for (int c = 0; c < R; ++c) {
        int piv = -1;
        double best = 0.0;
        for (int r = c; r < R; ++r) {
            double v = std::fabs(B[(size_t)r * R + c]);
            if (v > best) { best = v; piv = r; }
        }
        if (piv < 0 || best < 1e-11) return false;
        if (piv != c) {
            for (int k = 0; k < R; ++k) {
                std::swap(B[(size_t)piv * R + k], B[(size_t)c * R + k]);
                std::swap(inv[(size_t)piv * R + k], inv[(size_t)c * R + k]);
            }
        }
        double d = 1.0 / B[(size_t)c * R + c];
        for (int k = 0; k < R; ++k) {
            B[(size_t)c * R + k] *= d;
            inv[(size_t)c * R + k] *= d;
        }
        for (int r = 0; r < R; ++r) {
            if (r == c) continue;
            double f = B[(size_t)r * R + c];
            if (f == 0.0) continue;
            for (int k = 0; k < R; ++k) {
                B[(size_t)r * R + k] -= f * B[(size_t)c * R + k];
                inv[(size_t)r * R + k] -= f * inv[(size_t)c * R + k];
            }
        }
    }
    Binv_ = std::move(inv);
    xB_.assign(R, 0.0);
    for (int r = 0; r < R; ++r) {
        double s = 0.0;
        const double* row = &Binv_[(size_t)r * R];
        for (int i = 0; i < R; ++i) s += row[i] * b_[i];
        xB_[r] = s < 0.0 ? (s > -1e-7 ? 0.0 : s) : s;
    }
    y_.assign(R, 0.0);
    since_refactor_ = 0;
    compute_duals();
    return true;
}

DenseSimplex::Status DenseSimplex::solve(long max_iters, std::mt19937& rng) {
    const int R = R_;
    const double rc_tol = 1e-9;
    const double feas_tol = 1e-9;
    const double piv_tol = 1e-9;
    std::vector<double> alpha(R);
    long degenerate_streak = 0;
    bool bland = false;
    for (long it = 0; it < max_iters; ++it) {
        // entering variable
        int q = -1;
        double best = -rc_tol;
        int ties = 0;
        const int ncols = (int)cols_.size();
        for (int k = 0; k < ncols; ++k) {
            const Col& c = cols_[k];
            if (!c.active || pos_[k] >= 0) continue;
            double d = c.c;
            for (const auto& e : c.a) d -= y_[e.first] * e.second;
            if (bland) {
                if (d < -rc_tol) { q = k; break; }
            } else if (d < best - 1e-12) {
                best = d; q = k; ties = 1;
            } else if (d < -rc_tol && std::fabs(d - best) <= 1e-12) {
                // random tie-breaking against cycling
                ++ties;
                if (std::uniform_int_distribution<int>(0, ties - 1)(rng) == 0) q = k;
            }
        }
        if (q < 0) return OPTIMAL;

        // direction alpha = Binv * A_q
        std::fill(alpha.begin(), alpha.end(), 0.0);
        for (const auto& e : cols_[q].a) {
            const double a = e.second;
            const int i = e.first;
            for (int r = 0; r < R; ++r) alpha[r] += Binv_[(size_t)r * R + i] * a;
        }
        // Harris two-pass ratio test
        double theta_max = kInf;
        for (int r = 0; r < R; ++r) {
            if (alpha[r] > piv_tol) {
                double t = (xB_[r] + feas_tol) / alpha[r];
                if (t < theta_max) theta_max = t;
            }
        }
        if (theta_max == kInf) return UNBOUNDED;
        int leave = -1;
        double best_alpha = 0.0;
        for (int r = 0; r < R; ++r) {
            if (alpha[r] > piv_tol) {
                double t = xB_[r] / alpha[r];
                if (t <= theta_max && alpha[r] > best_alpha) {
                    best_alpha = alpha[r];
                    leave = r;
                }
            }
        }
        if (leave < 0) return UNBOUNDED;
        double theta = xB_[leave] / alpha[leave];
        if (theta < 0.0) theta = 0.0;
        if (theta <= 1e-12) {
            if (++degenerate_streak > 40L * R + 200) bland = true;
        } else {
            degenerate_streak = 0;
        }
        for (int r = 0; r < R; ++r) {
            xB_[r] -= theta * alpha[r];
            if (xB_[r] < 0.0 && xB_[r] > -1e-7) xB_[r] = 0.0;
        }
        xB_[leave] = theta;
        pos_[basis_[leave]] = -1;
        basis_[leave] = q;
        pos_[q] = leave;
        // rank-one update of the inverse
        const double piv = alpha[leave];
        double* prow = &Binv_[(size_t)leave * R];
        for (int k = 0; k < R; ++k) prow[k] /= piv;
        for (int r = 0; r < R; ++r) {
            if (r == leave || alpha[r] == 0.0) continue;
            const double f = alpha[r];
            double* row = &Binv_[(size_t)r * R];
            for (int k = 0; k < R; ++k) row[k] -= f * prow[k];
        }
        if (++since_refactor_ >= 50) {
            if (!refactor()) return SINGULAR;
        } else {
            compute_duals();
        }
    }
    return ITER_LIMIT;
}

// --------------------------------------------------------------------------
// Pricing: exact min-max knapsack
//
//   min over S (forced ⊆ S ⊆ forced ∪ cand, load(S) <= Q)
//       max_r  s_r(S)  -  mu,        s_r(S) = a_r + sum_{j in S} (p_rj - price_j)
//
// (scenario 0 is the implicit zero cut, theta >= 0).  Solved by scenario
// generation: branch-and-bound over a small active set A of scenarios (whose
// minimum is a lower bound on the true min-max) with bounds from single
// scenarios and from a convex combination of A optimised at the root
// (Lagrangian relaxation of the min-max).  When the optimum under A also
// attains its maximum inside A it is optimal for every scenario; otherwise the
// violated scenario joins A.  Every enumerated set is evaluated against all
// scenarios, so the incumbent is always a true reduced cost.
// --------------------------------------------------------------------------

struct PricingOut {
    double best_rc = kInf;          // best reduced cost of a non-empty feasible set
    double min_rc_lb = -kInf;       // valid lower bound on the pricing minimum
    bool exact = false;
    bool forced_infeasible = false; // forced customers exceed the capacity
    std::vector<std::pair<double, u64>> found;  // rc < -eps, best first
    long nodes = 0;
};

class Pricer {
public:
    Pricer(const Problem& P, int v) : P_(P), vh_(P.veh[v]) {}

    PricingOut run(u64 forced_mask, u64 cand_mask,
                   const std::vector<double>& price, double mu,
                   double eps, double seed_rc, long node_limit, int max_found,
                   const Clock& clock);

private:
    // A bounding scenario: a real cut or a convex combination of cuts.
    struct Bound {
        std::vector<double> d;      // value per candidate (index into cand_)
        double base = 0.0;          // value at the forced set
        std::vector<int> order;     // negative-value candidates by density
        double acc = 0.0;           // running sum over included items (DFS)
    };

    void sort_by_density(std::vector<int>& order, const std::vector<double>& d) const;
    double frac_knap(const Bound& b, int idx, i64 cap_left) const;
    void consider(u64 chosen);
    void dfs(int idx, i64 load, u64 chosen);
    void build_bounds(const std::vector<int>& A);
    void optimise_lambda(const std::vector<int>& A, std::vector<double>& lam) const;

    const Problem& P_;
    const Vehicle& vh_;

    // per call
    int K_ = 0;                            // scenarios incl. the zero cut (index 0)
    int nc_ = 0;                           // number of candidates
    std::vector<int> cand_;                // candidate -> customer
    std::vector<i64> q_;                   // candidate volume
    std::vector<std::vector<double>> d_;   // K x nc  scenario values
    std::vector<double> s_;                // K  scenario values at the forced set
    std::vector<double> cur_;              // K  running scenario values (DFS)
    std::vector<Bound> bounds_;            // active scenarios + convex combination
    std::vector<int> sub_;                 // candidates that help under A
    std::vector<int> order_;               // DFS order (subset of candidates)
    std::vector<int> pos_in_order_;        // candidate -> DFS position (nc_ when unused)
    std::vector<int> A_;                   // active scenarios
    std::vector<char> in_A_;
    i64 cap_rem_ = 0;
    u64 forced_mask_ = 0;
    double mu_ = 0.0;
    double eps_ = 0.0;
    double prune_eps_ = 1e-9;
    double best_rc_ = kInf;                // true reduced cost incumbent
    double round_min_ = kInf;              // min A-value seen this round
    u64 round_set_ = 0;
    double frontier_lb_ = kInf;            // min bound of subtrees left unexplored
    long nodes_ = 0;
    long node_limit_ = 0;
    bool aborted_ = false;
    int max_found_ = 0;
    const Clock* clock_ = nullptr;
    std::vector<std::pair<double, u64>> found_;
    mutable std::vector<std::pair<double, int>> scratch_;
};

void Pricer::sort_by_density(std::vector<int>& order, const std::vector<double>& d) const {
    std::sort(order.begin(), order.end(), [&](int x, int y) {
        // most negative value per unit volume first; zero-volume items first
        if (q_[x] == 0 && q_[y] == 0) return d[x] < d[y];
        if (q_[x] == 0) return true;
        if (q_[y] == 0) return false;
        return d[x] * (double)q_[y] < d[y] * (double)q_[x];
    });
}

double Pricer::frac_knap(const Bound& b, int idx, i64 cap_left) const {
    // Zero-volume items are sorted first, so once the capacity is exhausted
    // every later item has positive volume and the greedy fill stops.
    double val = 0.0;
    for (int c : b.order) {
        if (pos_in_order_[c] < idx) continue;   // already decided (or outside sub_)
        const i64 qc = q_[c];
        if (qc <= cap_left) {
            val += b.d[c];
            cap_left -= qc;
        } else {
            if (cap_left > 0) val += b.d[c] * ((double)cap_left / (double)qc);
            break;
        }
    }
    return val;
}

void Pricer::consider(u64 chosen) {
    // exact value with every scenario
    double mx = cur_[0];
    for (int r = 1; r < K_; ++r) if (cur_[r] > mx) mx = cur_[r];
    const double rc = mx - mu_;
    if (rc < best_rc_) best_rc_ = rc;
    if (rc < -eps_) {
        const u64 mask = chosen | forced_mask_;
        auto pos = std::lower_bound(found_.begin(), found_.end(), rc,
                                    [](const auto& e, double val) { return e.first < val; });
        if ((int)found_.size() < max_found_ || pos != found_.end()) {
            found_.insert(pos, {rc, mask});
            if ((int)found_.size() > max_found_) found_.pop_back();
        }
    }
    // value under the active set only
    double ma = -kInf;
    for (int r : A_) if (cur_[r] > ma) ma = cur_[r];
    if (ma < round_min_) { round_min_ = ma; round_set_ = chosen; }
}

void Pricer::dfs(int idx, i64 load, u64 chosen) {
    if (aborted_) return;
    ++nodes_;
    if ((nodes_ & 1023) == 0 && clock_ && clock_->expired()) aborted_ = true;
    if (nodes_ > node_limit_) aborted_ = true;
    // Prune against the true incumbent: the A-objective never exceeds the
    // true objective, so a subtree whose A-bound reaches the incumbent holds
    // no better set.
    const double prune_at = best_rc_ + mu_ - prune_eps_;
    const i64 cap_left = cap_rem_ - load;
    double lb = -kInf;
    for (const Bound& b : bounds_) {
        const double v = b.acc + frac_knap(b, idx, cap_left);
        if (v > lb) {
            lb = v;
            if (lb >= prune_at) return;
        }
    }
    if (aborted_) {
        if (lb < frontier_lb_) frontier_lb_ = lb;
        return;
    }
    if (idx >= (int)order_.size()) return;
    const int c = order_[idx];
    const i64 qc = q_[c];
    if (qc <= cap_left) {
        for (int r = 0; r < K_; ++r) cur_[r] += d_[r][c];
        for (Bound& b : bounds_) b.acc += b.d[c];
        const u64 chosen2 = chosen | (1ULL << cand_[c]);
        consider(chosen2);
        dfs(idx + 1, load + qc, chosen2);
        for (int r = 0; r < K_; ++r) cur_[r] -= d_[r][c];
        for (Bound& b : bounds_) b.acc -= b.d[c];
        if (aborted_) {
            // the exclude branch below is left unexplored; its bound is >= lb
            if (lb < frontier_lb_) frontier_lb_ = lb;
            return;
        }
    }
    dfs(idx + 1, load, chosen);
}

void Pricer::optimise_lambda(const std::vector<int>& A, std::vector<double>& lam) const {
    // maximise phi(lam) = sum_r lam_r s_r + FK(sum_r lam_r d_r) over the simplex
    // (concave, piecewise linear) by entropic mirror ascent; keep the best.
    const int a = (int)A.size();
    std::vector<double> best_lam = lam;
    double best_phi = -kInf;
    std::vector<double> e(nc_), x(nc_), g(a);
    std::vector<int> ord;
    for (int it = 0; it < 24; ++it) {
        ord.clear();
        for (int c : sub_) {
            double v = 0.0;
            for (int k = 0; k < a; ++k) v += lam[k] * d_[A[k]][c];
            e[c] = v;
            if (v < 0.0) ord.push_back(c);
        }
        sort_by_density(ord, e);
        std::fill(x.begin(), x.end(), 0.0);
        double phi = 0.0;
        for (int k = 0; k < a; ++k) phi += lam[k] * s_[A[k]];
        i64 cl = cap_rem_;
        for (int c : ord) {
            const i64 qc = q_[c];
            if (qc <= cl) { phi += e[c]; x[c] = 1.0; cl -= qc; }
            else { if (cl > 0) { x[c] = (double)cl / (double)qc; phi += e[c] * x[c]; } break; }
        }
        if (phi > best_phi) { best_phi = phi; best_lam = lam; }
        double gmax = 0.0;
        for (int k = 0; k < a; ++k) {
            double v = s_[A[k]];
            for (int c : ord) if (x[c] > 0.0) v += d_[A[k]][c] * x[c];
            g[k] = v;
            gmax = std::max(gmax, std::fabs(v));
        }
        if (gmax <= 0.0) break;
        const double eta = 1.5 / (gmax * std::sqrt((double)(it + 1)));
        double z = 0.0;
        for (int k = 0; k < a; ++k) { lam[k] *= std::exp(eta * g[k]); z += lam[k]; }
        if (!(z > 0.0) || !std::isfinite(z)) break;
        for (int k = 0; k < a; ++k) lam[k] /= z;
    }
    lam = best_lam;
}

void Pricer::build_bounds(const std::vector<int>& A) {
    // candidates that lower some active scenario
    sub_.clear();
    for (int c = 0; c < nc_; ++c) {
        bool helps = false;
        for (int r : A) if (d_[r][c] < 0.0) { helps = true; break; }
        if (helps) sub_.push_back(c);
    }
    bounds_.clear();
    for (int r : A) {
        Bound b;
        b.d = d_[r];
        b.base = s_[r];
        for (int c : sub_) if (d_[r][c] < 0.0) b.order.push_back(c);
        sort_by_density(b.order, b.d);
        bounds_.push_back(std::move(b));
    }
    std::vector<double> key(nc_, 0.0);
    if (A.size() >= 2) {
        std::vector<double> lam(A.size(), 1.0 / (double)A.size());
        optimise_lambda(A, lam);
        Bound b;
        b.d.assign(nc_, 0.0);
        b.base = 0.0;
        for (size_t k = 0; k < A.size(); ++k) {
            b.base += lam[k] * s_[A[k]];
            for (int c = 0; c < nc_; ++c) b.d[c] += lam[k] * d_[A[k]][c];
        }
        for (int c : sub_) if (b.d[c] < 0.0) b.order.push_back(c);
        sort_by_density(b.order, b.d);
        key = b.d;
        bounds_.push_back(std::move(b));
    } else {
        key = d_[A[0]];
    }
    // DFS order: density under the combined scenario, most negative first
    order_ = sub_;
    for (int c : order_) key[c] = key[c] / (double)std::max<i64>(1, q_[c]);
    std::sort(order_.begin(), order_.end(), [&](int x, int y) { return key[x] < key[y]; });
    pos_in_order_.assign(nc_, nc_);
    for (int i = 0; i < (int)order_.size(); ++i) pos_in_order_[order_[i]] = i;
    for (Bound& b : bounds_) b.acc = b.base;
}

PricingOut Pricer::run(u64 forced_mask, u64 cand_mask,
                       const std::vector<double>& price, double mu,
                       double eps, double seed_rc, long node_limit, int max_found,
                       const Clock& clock) {
    PricingOut out;
    const int n = P_.n;
    const int Kc = (int)vh_.cuts.size();
    K_ = Kc + 1;
    mu_ = mu;
    eps_ = eps;
    forced_mask_ = forced_mask;
    node_limit_ = node_limit;
    max_found_ = std::max(1, max_found);
    clock_ = &clock;
    nodes_ = 0;
    aborted_ = false;
    frontier_lb_ = kInf;
    found_.clear();
    best_rc_ = seed_rc;

    i64 forced_load = mask_sum(P_.q, forced_mask);
    if (forced_load > vh_.cap) {
        out.forced_infeasible = true;
        out.exact = true;
        out.best_rc = kInf;
        out.min_rc_lb = kInf;
        return out;
    }
    cap_rem_ = vh_.cap - forced_load;

    // scenario values at the forced set
    s_.assign(K_, 0.0);
    for (u64 mm = forced_mask; mm; mm &= mm - 1) s_[0] -= price[__builtin_ctzll(mm)];
    for (int r = 0; r < Kc; ++r) {
        double val = vh_.cuts[r].a;
        for (u64 mm = forced_mask; mm; mm &= mm - 1) {
            int j = __builtin_ctzll(mm);
            val += vh_.cuts[r].p[j] - price[j];
        }
        s_[r + 1] = val;
    }

    // Candidates.  An item that lowers no scenario can be removed from any set
    // with at least one other item without increasing the objective, so it is
    // never part of an optimal set of size >= 2.  Its singleton is the only
    // exception when nothing is forced; it is evaluated explicitly below.
    cand_.clear();
    q_.clear();
    std::vector<int> singletons;
    for (int j = 0; j < n; ++j) {
        if (!((cand_mask >> j) & 1ULL)) continue;
        if (P_.q[j] > cap_rem_) continue;
        bool helps = -price[j] < 0.0;
        for (int r = 0; r < Kc && !helps; ++r) if (vh_.cuts[r].p[j] - price[j] < 0.0) helps = true;
        if (!helps) {
            if (forced_mask == 0) singletons.push_back(j);
            continue;
        }
        cand_.push_back(j);
        q_.push_back(P_.q[j]);
    }
    nc_ = (int)cand_.size();
    d_.assign(K_, std::vector<double>(nc_, 0.0));
    for (int c = 0; c < nc_; ++c) {
        const int j = cand_[c];
        d_[0][c] = -price[j];
        for (int r = 0; r < Kc; ++r) d_[r + 1][c] = vh_.cuts[r].p[j] - price[j];
    }

    // Floating-point allowance for the bounds: every partial sum inside the
    // search has at most nc_ + 2 terms whose magnitude is bounded by scale.
    double scale = std::fabs(mu);
    for (int r = 0; r < K_; ++r) scale = std::max(scale, std::fabs(s_[r]));
    {
        double col = 0.0;
        for (int c = 0; c < nc_; ++c) {
            double mx = 0.0;
            for (int r = 0; r < K_; ++r) mx = std::max(mx, std::fabs(d_[r][c]));
            col += mx;
        }
        scale += col;
    }
    const double fp_margin = 8.0 * (double)(nc_ + 4) * scale * 1.1102230246251565e-16;
    prune_eps_ = 1e-9 + fp_margin;

    // Initial active set: the scenario binding at the forced set and at a
    // greedy incumbent, plus the strongest root single-scenario bound.
    A_.clear();
    in_A_.assign(K_, 0);
    auto activate = [&](int r) {
        if (!in_A_[r]) { in_A_[r] = 1; A_.push_back(r); }
    };
    cur_ = s_;
    {
        int arg = 0;
        for (int r = 1; r < K_; ++r) if (s_[r] > s_[arg]) arg = r;
        activate(arg);
    }
    if (forced_mask != 0) { round_min_ = kInf; consider(0); }
    for (int j : singletons) {
        std::vector<double> save = cur_;
        cur_[0] = -price[j];
        for (int r = 0; r < Kc; ++r) cur_[r + 1] = vh_.cuts[r].a + vh_.cuts[r].p[j] - price[j];
        consider(1ULL << j);
        cur_ = save;
    }
    {
        // greedy incumbent along the average density (include while the max drops)
        std::vector<int> ord(nc_);
        std::iota(ord.begin(), ord.end(), 0);
        std::vector<double> key(nc_, 0.0);
        for (int c = 0; c < nc_; ++c) {
            double acc = 0.0;
            for (int r = 0; r < K_; ++r) acc += std::min(0.0, d_[r][c]);
            key[c] = acc / (double)std::max<i64>(1, q_[c]);
        }
        std::sort(ord.begin(), ord.end(), [&](int x, int y) { return key[x] < key[y]; });
        i64 load = 0;
        u64 chosen = 0;
        double curmax = -kInf;
        for (int r = 0; r < K_; ++r) curmax = std::max(curmax, cur_[r]);
        for (int c : ord) {
            if (load + q_[c] > cap_rem_) continue;
            double mx = -kInf;
            for (int r = 0; r < K_; ++r) mx = std::max(mx, cur_[r] + d_[r][c]);
            if (mx < curmax) {
                for (int r = 0; r < K_; ++r) cur_[r] += d_[r][c];
                load += q_[c];
                chosen |= 1ULL << cand_[c];
                curmax = mx;
                consider(chosen);
            }
        }
        int arg = 0;
        for (int r = 1; r < K_; ++r) if (cur_[r] > cur_[arg]) arg = r;
        activate(arg);
        cur_ = s_;
    }
    {
        int best_r = -1;
        double best_v = -kInf;
        std::vector<int> ord;
        for (int r = 0; r < K_; ++r) {
            Bound b;
            b.d = d_[r];
            for (int c = 0; c < nc_; ++c) if (d_[r][c] < 0.0) b.order.push_back(c);
            sort_by_density(b.order, b.d);
            pos_in_order_.assign(nc_, nc_);
            for (int i = 0; i < (int)b.order.size(); ++i) pos_in_order_[b.order[i]] = i;
            // every candidate is "undecided" at the root: position >= 0
            const double v = s_[r] + frac_knap(b, 0, cap_rem_);
            if (v > best_v) { best_v = v; best_r = r; }
        }
        if (best_r >= 0) activate(best_r);
    }

    // Scenario generation rounds.
    double proved_lb = -kInf;      // LB on the true objective from completed rounds
    double round_lb_aborted = kInf;
    while (true) {
        build_bounds(A_);
        // the forced set alone is a member of the search space
        round_min_ = kInf;
        if (forced_mask != 0) {
            round_min_ = -kInf;
            for (int r : A_) round_min_ = std::max(round_min_, s_[r]);
        }
        round_set_ = 0;
        frontier_lb_ = kInf;
        cur_ = s_;
        dfs(0, 0, 0);
        const double prune_at = best_rc_ + mu_ - prune_eps_;
        if (aborted_) {
            round_lb_aborted = std::min({frontier_lb_, round_min_, prune_at});
            break;
        }
        if (!(round_min_ < prune_at)) {
            // every set has A-value >= incumbent: the incumbent is optimal
            proved_lb = std::max(proved_lb, prune_at);
            break;
        }
        proved_lb = std::max(proved_lb, round_min_);
        // true value of the A-optimal set
        double mx = -kInf;
        int arg = 0;
        {
            std::vector<double> val = s_;
            for (u64 mm = round_set_; mm; mm &= mm - 1) {
                const int j = __builtin_ctzll(mm);
                // map customer -> candidate index
                int c = -1;
                for (int k = 0; k < nc_; ++k) if (cand_[k] == j) { c = k; break; }
                if (c < 0) continue;
                for (int r = 0; r < K_; ++r) val[r] += d_[r][c];
            }
            for (int r = 0; r < K_; ++r) if (val[r] > mx) { mx = val[r]; arg = r; }
        }
        if (mx <= round_min_ + prune_eps_ || (int)A_.size() >= K_) {
            // the A-optimum is optimal for every scenario (its true value was
            // already recorded by consider)
            break;
        }
        activate(arg);
    }

    out.nodes = nodes_;
    out.best_rc = best_rc_;
    out.found = std::move(found_);
    out.exact = !aborted_;
    if (out.exact) {
        if (std::isfinite(best_rc_)) out.min_rc_lb = best_rc_ - prune_eps_ - fp_margin;
        else out.min_rc_lb = kInf;   // no feasible non-empty set at all
    } else {
        double lb = std::min(best_rc_, std::max(proved_lb, round_lb_aborted) - mu_);
        out.min_rc_lb = std::isfinite(lb) ? lb - prune_eps_ - fp_margin : lb;
    }
    return out;
}

// --------------------------------------------------------------------------
// Solutions and local search (families of sets, canonical inside a type)
// --------------------------------------------------------------------------

struct Solution {
    std::vector<u64> set;   // per vehicle
    double cost = kInf;
    bool valid = false;
};

class Evaluator {
public:
    explicit Evaluator(const Problem& P) : P_(P), cache_(P.m) {}

    double theta(int v, u64 mask) {
        if (mask == 0) return P_.veh[v].empty_theta;
        auto& mp = cache_[v];
        auto it = mp.find(mask);
        if (it != mp.end()) return it->second;
        double val = theta_mask(P_.veh[v], mask);
        if (mp.size() > 400000) mp.clear();
        mp.emplace(mask, val);
        return val;
    }

    // Sort sets inside every group by score (descending, stable) and return
    // the total cost, or +inf when a set does not fit its vehicle.
    double canonical_cost(std::vector<u64>& set, u64 active_mask) {
        std::vector<std::pair<i64, u64>> tmp;
        double cost = 0.0;
        u64 served = 0;
        for (const auto& g : P_.groups) {
            tmp.clear();
            for (int v : g) tmp.emplace_back(mask_sum(P_.w, set[v]), set[v]);
            std::stable_sort(tmp.begin(), tmp.end(),
                             [](const auto& x, const auto& y) { return x.first > y.first; });
            for (size_t i = 0; i < g.size(); ++i) {
                const int v = g[i];
                set[v] = tmp[i].second;
                if (mask_sum(P_.q, set[v]) > P_.veh[v].cap) return kInf;
                served |= set[v];
                cost += theta(v, set[v]);
            }
        }
        for (u64 mm = active_mask & ~served; mm; mm &= mm - 1) cost += P_.c_out[__builtin_ctzll(mm)];
        return cost;
    }

    bool improve(Solution& sol, u64 active_mask, const Clock& clock, double time_cap) {
        // first-improvement relocate / swap on the canonical family
        const int n = P_.n, m = P_.m;
        bool any = false;
        bool improved = true;
        const double t0 = clock.elapsed();
        std::vector<int> owner(n, -1);
        while (improved) {
            improved = false;
            if (clock.elapsed() - t0 > time_cap || clock.expired()) break;
            std::fill(owner.begin(), owner.end(), -1);
            for (int v = 0; v < m; ++v)
                for (u64 mm = sol.set[v]; mm; mm &= mm - 1) owner[__builtin_ctzll(mm)] = v;
            // relocations (including to/from outsourcing)
            for (int j = 0; j < n && !improved; ++j) {
                if (!((active_mask >> j) & 1ULL)) continue;
                const int from = owner[j];
                for (int to = -1; to < m; ++to) {
                    if (to == from) continue;
                    if (to >= 0 && mask_sum(P_.q, sol.set[to]) + P_.q[j] > P_.veh[to].cap) continue;
                    std::vector<u64> cand = sol.set;
                    if (from >= 0) cand[from] &= ~(1ULL << j);
                    if (to >= 0) cand[to] |= 1ULL << j;
                    double c = canonical_cost(cand, active_mask);
                    if (c < sol.cost - 1e-9) {
                        sol.set = std::move(cand);
                        sol.cost = c;
                        improved = any = true;
                        break;
                    }
                }
            }
            if (improved) continue;
            // swaps between two vehicles
            for (int j = 0; j < n && !improved; ++j) {
                if (owner[j] < 0) continue;
                for (int k = j + 1; k < n && !improved; ++k) {
                    if (owner[k] < 0 || owner[k] == owner[j]) continue;
                    const int a = owner[j], b = owner[k];
                    if (mask_sum(P_.q, sol.set[a]) - P_.q[j] + P_.q[k] > P_.veh[a].cap) continue;
                    if (mask_sum(P_.q, sol.set[b]) - P_.q[k] + P_.q[j] > P_.veh[b].cap) continue;
                    std::vector<u64> cand = sol.set;
                    cand[a] = (cand[a] & ~(1ULL << j)) | (1ULL << k);
                    cand[b] = (cand[b] & ~(1ULL << k)) | (1ULL << j);
                    double c = canonical_cost(cand, active_mask);
                    if (c < sol.cost - 1e-9) {
                        sol.set = std::move(cand);
                        sol.cost = c;
                        improved = any = true;
                    }
                }
            }
        }
        return any;
    }

private:
    const Problem& P_;
    std::vector<std::unordered_map<u64, double>> cache_;
};

// --------------------------------------------------------------------------
// Branch-and-price
// --------------------------------------------------------------------------

struct Params {
    double time_limit = 60.0;
    double rel_gap = 1e-4;
    double abs_gap = 1e-9;
    double bound_stop = kInf;
    long max_nodes = 2000000;
    long pricing_node_limit = 300000;
    int max_cols_per_pricing = 6;
    long simplex_iter_limit = 200000;
    int verbose = 0;
    unsigned seed = 12345u;
};

struct Result {
    std::vector<int> assign;
    double ub = kInf;
    double lb = -kInf;
    bool optimal = false;
    bool has_solution = false;
    std::string status = "unknown";
    long nodes = 0;
    long cg_iterations = 0;
    long pricing_nodes = 0;
    long columns = 0;
    double root_lb = -kInf;
    double seconds = 0.0;
    std::string error;
};

struct PoolCol {
    int v;
    u64 mask;
    double cost;
    double score;   // as double for the LP rows
};

struct BBNode {
    double lb;
    int depth;
    long id;
    std::vector<int8_t> fix;    // n*m: 0 free, 1 forced, -1 forbidden
    std::vector<int8_t> yfix;   // m: 0 free, 1 used, -1 unused
    bool operator<(const BBNode& o) const {
        // priority_queue is a max-heap: smaller lb first, then deeper first
        if (lb != o.lb) return lb > o.lb;
        return depth < o.depth;
    }
};

class BranchAndPrice {
public:
    BranchAndPrice(const Problem& P, const Params& prm, const std::vector<int>& hint)
        : P_(P), prm_(prm), hint_(hint), eval_(P), clock_(prm.time_limit),
          rng_(prm.seed) {
        for (int j = 0; j < P_.n; ++j) active_mask_ |= 1ULL << j;
        pricers_.reserve(P_.m);
        for (int v = 0; v < P_.m; ++v) pricers_.emplace_back(P_, v);
    }
    Result solve();

private:
    struct NodeLP {
        DenseSimplex* lp = nullptr;
        std::vector<int> z_col;        // n (or -1)
        std::vector<int> empty_col;    // m (or -1)
        std::vector<int> art_col;      // rows n+m (or -1)
        std::vector<int> pool_to_col;  // pool id -> column id (-1 absent)
        std::vector<int> col_to_pool;  // column id -> pool id (-1 for structural)
        std::vector<u64> forbid;       // m
        std::vector<u64> force;        // m
        std::vector<int> basis;
    };

    int add_pool_column(int v, u64 mask);
    // 0 = built, 1 = node infeasible by fixings, 2 = LP failure
    int build_lp(const BBNode& node, NodeLP& L);
    void add_pool_to_lp(NodeLP& L, int pid);
    // Returns node lower bound (kInf when infeasible); fills lambda solution.
    double solve_node(const BBNode& node, NodeLP& L, bool& infeasible);
    void update_incumbent(Solution& sol, const char* source);
    void hint_heuristic();
    void rounding_heuristic(const NodeLP& L);
    void lp_fractional(const NodeLP& L, std::vector<double>& alpha, std::vector<double>& yv);
    double stop_gap() const {
        return std::max(prm_.abs_gap, prm_.rel_gap * std::fabs(best_.cost));
    }
    void log(const char* fmt, ...) const;

    const Problem& P_;
    Params prm_;
    std::vector<int> hint_;
    Evaluator eval_;
    Clock clock_;
    std::mt19937 rng_;
    u64 active_mask_ = 0;
    std::vector<Pricer> pricers_;
    std::vector<PoolCol> pool_;
    std::unordered_set<u64> pool_keys_;   // (v << 58) ^ mask... use combined key
    Solution best_;
    Result res_;
    double big_m_ = 1.0;
    double pruned_lb_min_ = kInf;
};

void BranchAndPrice::log(const char* fmt, ...) const {
    if (prm_.verbose <= 0) return;
    va_list args;
    va_start(args, fmt);
    std::fprintf(stdout, "[s2bp %.2fs] ", clock_.elapsed());
    std::vfprintf(stdout, fmt, args);
    std::fprintf(stdout, "\n");
    std::fflush(stdout);
    va_end(args);
}

static inline u64 pool_key(int v, u64 mask) {
    // n <= 64 uses all 64 mask bits; combine with the vehicle through a hash mix
    u64 h = mask * 0x9E3779B97F4A7C15ULL;
    h ^= (u64)(v + 1) * 0xC2B2AE3D27D4EB4FULL;
    h ^= h >> 29;
    return h;
}

int BranchAndPrice::add_pool_column(int v, u64 mask) {
    if (mask == 0) return -1;
    const u64 key = pool_key(v, mask);
    if (!pool_keys_.insert(key).second) {
        // possible hash collision or a genuine duplicate: verify by scan of recent columns
        for (int i = (int)pool_.size() - 1; i >= 0; --i)
            if (pool_[i].v == v && pool_[i].mask == mask) return -1;
        // collision with a different column: fall through and add anyway
    }
    PoolCol pc;
    pc.v = v;
    pc.mask = mask;
    pc.cost = eval_.theta(v, mask);
    pc.score = 0.0;
    for (u64 mm = mask; mm; mm &= mm - 1) pc.score += P_.w_lp[__builtin_ctzll(mm)];
    pool_.push_back(pc);
    return (int)pool_.size() - 1;
}

void BranchAndPrice::add_pool_to_lp(NodeLP& L, int pid) {
    const PoolCol& pc = pool_[pid];
    const int n = P_.n, m = P_.m;
    DenseSimplex::Col col;
    col.c = pc.cost;
    for (u64 mm = pc.mask; mm; mm &= mm - 1) col.a.emplace_back(__builtin_ctzll(mm), 1.0);
    col.a.emplace_back(n + pc.v, 1.0);
    if (P_.pair_first_of[pc.v] >= 0) col.a.emplace_back(n + m + P_.pair_first_of[pc.v], pc.score);
    if (P_.pair_second_of[pc.v] >= 0) col.a.emplace_back(n + m + P_.pair_second_of[pc.v], -pc.score);
    int cid = L.lp->add(std::move(col));
    if ((int)L.pool_to_col.size() < (int)pool_.size()) L.pool_to_col.resize(pool_.size(), -1);
    L.pool_to_col[pid] = cid;
    if ((int)L.col_to_pool.size() <= cid) L.col_to_pool.resize(cid + 1, -1);
    L.col_to_pool[cid] = pid;
}

int BranchAndPrice::build_lp(const BBNode& node, NodeLP& L) {
    const int n = P_.n, m = P_.m, Pn = (int)P_.pairs.size();
    const int R = n + m + Pn;
    std::vector<double> b(R, 0.0);
    for (int i = 0; i < n + m; ++i) b[i] = 1.0;
    delete L.lp;
    L.lp = new DenseSimplex(R, b);
    L.z_col.assign(n, -1);
    L.empty_col.assign(m, -1);
    L.art_col.assign(n + m, -1);
    L.pool_to_col.assign(pool_.size(), -1);
    L.col_to_pool.clear();
    L.forbid.assign(m, 0);
    L.force.assign(m, 0);
    std::vector<char> z_allowed(n, 1);
    for (int j = 0; j < n; ++j) {
        for (int v = 0; v < m; ++v) {
            const int8_t f = node.fix[(size_t)j * m + v];
            if (f == 1) { L.force[v] |= 1ULL << j; z_allowed[j] = 0; }
            else if (f == -1) L.forbid[v] |= 1ULL << j;
        }
    }
    for (int v = 0; v < m; ++v) {
        if (node.yfix[v] == -1) L.forbid[v] = active_mask_;
        // customers that do not fit are never candidates
        for (int j = 0; j < n; ++j) if (P_.q[j] > P_.veh[v].cap) L.forbid[v] |= 1ULL << j;
        if (L.force[v] & L.forbid[v]) return 1;   // contradictory fixings
        if (mask_sum(P_.q, L.force[v]) > P_.veh[v].cap) return 1;
    }
    std::vector<int> basis(R, -1);
    for (int j = 0; j < n; ++j) {
        if (z_allowed[j]) {
            DenseSimplex::Col c;
            c.c = P_.c_out[j];
            c.a.emplace_back(j, 1.0);
            L.z_col[j] = L.lp->add(std::move(c));
            basis[j] = L.z_col[j];
        } else {
            DenseSimplex::Col c;
            c.c = big_m_;
            c.a.emplace_back(j, 1.0);
            L.art_col[j] = L.lp->add(std::move(c));
            basis[j] = L.art_col[j];
        }
    }
    for (int v = 0; v < m; ++v) {
        if (node.yfix[v] != 1) {
            DenseSimplex::Col c;
            c.c = P_.veh[v].empty_theta;
            c.a.emplace_back(n + v, 1.0);
            L.empty_col[v] = L.lp->add(std::move(c));
            basis[n + v] = L.empty_col[v];
        } else {
            DenseSimplex::Col c;
            c.c = big_m_;
            c.a.emplace_back(n + v, 1.0);
            L.art_col[n + v] = L.lp->add(std::move(c));
            basis[n + v] = L.art_col[n + v];
        }
    }
    for (int p = 0; p < Pn; ++p) {
        DenseSimplex::Col c;
        c.c = 0.0;
        c.a.emplace_back(n + m + p, -1.0);
        basis[n + m + p] = L.lp->add(std::move(c));
    }
    L.col_to_pool.assign(L.lp->num_cols(), -1);
    for (int pid = 0; pid < (int)pool_.size(); ++pid) {
        const PoolCol& pc = pool_[pid];
        if (node.yfix[pc.v] == -1) continue;
        if (pc.mask & L.forbid[pc.v]) continue;
        if ((pc.mask & L.force[pc.v]) != L.force[pc.v]) continue;
        add_pool_to_lp(L, pid);
    }
    L.basis = basis;
    return L.lp->set_basis(basis) ? 0 : 2;
}

void BranchAndPrice::lp_fractional(const NodeLP& L, std::vector<double>& alpha,
                                   std::vector<double>& yv) {
    const int n = P_.n, m = P_.m;
    alpha.assign((size_t)n * m, 0.0);
    yv.assign(m, 0.0);
    const int ncols = L.lp->num_cols();
    for (int cid = 0; cid < ncols; ++cid) {
        const int pid = cid < (int)L.col_to_pool.size() ? L.col_to_pool[cid] : -1;
        if (pid < 0) continue;
        const double lam = L.lp->primal(cid);
        if (lam <= 1e-9) continue;
        const PoolCol& pc = pool_[pid];
        yv[pc.v] += lam;
        for (u64 mm = pc.mask; mm; mm &= mm - 1) alpha[(size_t)__builtin_ctzll(mm) * m + pc.v] += lam;
    }
}

double BranchAndPrice::solve_node(const BBNode& node, NodeLP& L, bool& infeasible) {
    infeasible = false;
    const int n = P_.n, m = P_.m, Pn = (int)P_.pairs.size();
    const int built = build_lp(node, L);
    if (built == 1) { infeasible = true; return kInf; }
    if (built == 2) { res_.error = "master LP basis is singular"; return -kInf; }
    double node_lb = -kInf;
    std::vector<double> price(n);
    int iter = 0;
    const double ub_target = best_.valid ? best_.cost - stop_gap() : kInf;
    while (true) {
        DenseSimplex::Status st = L.lp->solve(prm_.simplex_iter_limit, rng_);
        if (st == DenseSimplex::SINGULAR) {
            // rebuild from the identity basis and retry once
            DenseSimplex::Status st2 = DenseSimplex::SINGULAR;
            if (L.lp->set_basis(L.basis)) st2 = L.lp->solve(prm_.simplex_iter_limit, rng_);
            if (st2 != DenseSimplex::OPTIMAL) {
                res_.error = std::string("master LP failed after singular basis (restart status ")
                             + std::to_string((int)st2) + ", cols "
                             + std::to_string(L.lp->num_cols()) + ")";
                return node_lb;
            }
        } else if (st == DenseSimplex::UNBOUNDED) {
            res_.error = "master LP unbounded";
            return node_lb;
        } else if (st == DenseSimplex::ITER_LIMIT) {
            // keep going with the current duals: the Lagrangian bound is valid anyway
        }
        ++iter;
        ++res_.cg_iterations;
        std::vector<double> y = L.lp->duals();
        const double z = L.lp->objective();
        // ordering duals must be non-negative for the Lagrangian bound
        for (int p = 0; p < Pn; ++p) if (y[n + m + p] < 0.0) y[n + m + p] = 0.0;
        double lag = 0.0;
        double lag_scale = 0.0;
        for (int i = 0; i < n + m; ++i) { lag += y[i]; lag_scale += std::fabs(y[i]); }
        for (int j = 0; j < n; ++j) {
            if (L.z_col[j] >= 0) {
                double rc = P_.c_out[j] - y[j];
                lag_scale += std::fabs(P_.c_out[j]) + std::fabs(y[j]);
                if (rc < 0.0) lag += rc;
            }
        }
        const double rc_eps = std::max(1e-9, 1e-9 * (1.0 + std::fabs(z)));
        int added = 0;
        bool all_exact = true;
        for (int v = 0; v < m; ++v) {
            if (clock_.expired()) break;
            const double g = (P_.pair_first_of[v] >= 0 ? y[n + m + P_.pair_first_of[v]] : 0.0)
                           - (P_.pair_second_of[v] >= 0 ? y[n + m + P_.pair_second_of[v]] : 0.0);
            for (int j = 0; j < n; ++j) price[j] = y[j] + g * P_.w_lp[j];
            const double mu = y[n + v];
            // seed with the best reduced cost among this node's columns of v
            double seed = kInf;
            for (int pid = 0; pid < (int)pool_.size(); ++pid) {
                if (pool_[pid].v != v || L.pool_to_col[pid] < 0) continue;
                double rc = pool_[pid].cost - mu;
                for (u64 mm = pool_[pid].mask; mm; mm &= mm - 1) rc -= price[__builtin_ctzll(mm)];
                if (rc < seed) seed = rc;
            }
            const u64 cand = active_mask_ & ~L.forbid[v] & ~L.force[v];
            PricingOut out = pricers_[v].run(L.force[v], cand, price, mu, rc_eps, seed,
                                             prm_.pricing_node_limit,
                                             prm_.max_cols_per_pricing, clock_);
            res_.pricing_nodes += out.nodes;
            if (out.forced_infeasible) { infeasible = true; return kInf; }
            if (!out.exact) all_exact = false;
            double vterm = out.min_rc_lb;
            if (L.empty_col[v] >= 0) vterm = std::min(vterm, P_.veh[v].empty_theta - mu);
            if (vterm == kInf) {
                // vehicle has no admissible column at all
                infeasible = true;
                return kInf;
            }
            lag += vterm;
            lag_scale += std::fabs(vterm) + std::fabs(mu);
            for (const auto& f : out.found) {
                int pid = add_pool_column(v, f.second);
                if (pid >= 0) { add_pool_to_lp(L, pid); ++added; }
            }
        }
        if (clock_.expired()) {
            // partial pricing: the bound is only valid if every vehicle was priced
            return node_lb;
        }
        // rounding allowance for the (2n + 2m) term Lagrangian sum
        lag -= 8.0 * (double)(2 * n + 2 * m + 4) * lag_scale * 1.1102230246251565e-16;
        if (lag > node_lb) node_lb = lag;
        if (prm_.verbose >= 3)
            log("  cg it=%d z=%.6f lag=%.6f added=%d cols=%d", iter, z, lag, added, L.lp->num_cols());
        if (node_lb >= ub_target) return node_lb;
        if (added == 0) {
            if (!all_exact) {
                // could not certify convergence; bound is still valid, branch on the LP
                return node_lb;
            }
            // converged: check artificials
            for (int i = 0; i < n + m; ++i) {
                if (L.art_col[i] >= 0 && L.lp->primal(L.art_col[i]) > 1e-7) {
                    infeasible = true;
                    return kInf;
                }
            }
            return node_lb;
        }
    }
}

void BranchAndPrice::update_incumbent(Solution& sol, const char* source) {
    if (!std::isfinite(sol.cost)) return;
    // exact feasibility: capacity and order inside every group
    for (int v = 0; v < P_.m; ++v)
        if (mask_sum(P_.q, sol.set[v]) > P_.veh[v].cap) return;
    for (const auto& pr : P_.pairs)
        if (mask_sum(P_.w, sol.set[pr.first]) < mask_sum(P_.w, sol.set[pr.second])) return;
    if (!best_.valid || sol.cost < best_.cost - 1e-12) {
        best_ = sol;
        best_.valid = true;
        log("incumbent %.6f (%s)", best_.cost, source);
    }
}

void BranchAndPrice::hint_heuristic() {
    Solution sol;
    sol.set.assign(P_.m, 0);
    if ((int)hint_.size() == P_.n) {
        for (int j = 0; j < P_.n; ++j) {
            int v = hint_[j];
            if (v >= 0 && v < P_.m) sol.set[v] |= 1ULL << j;
        }
    }
    sol.cost = eval_.canonical_cost(sol.set, active_mask_);
    if (!std::isfinite(sol.cost)) {
        // over capacity somewhere: outsource everything as a start
        sol.set.assign(P_.m, 0);
        sol.cost = eval_.canonical_cost(sol.set, active_mask_);
    }
    update_incumbent(sol, "hint");
    eval_.improve(sol, active_mask_, clock_, 0.25 * prm_.time_limit);
    update_incumbent(sol, "hint+ls");
    // seed the pool with the incumbent's columns
    for (int v = 0; v < P_.m; ++v) add_pool_column(v, sol.set[v]);
}

void BranchAndPrice::rounding_heuristic(const NodeLP& L) {
    const int m = P_.m;
    // vehicle by vehicle, take the column with the largest lambda that does
    // not conflict with already taken customers; then repair by local search
    std::vector<std::pair<double, int>> cands;
    const int ncols = L.lp->num_cols();
    for (int cid = 0; cid < ncols; ++cid) {
        const int pid = cid < (int)L.col_to_pool.size() ? L.col_to_pool[cid] : -1;
        if (pid < 0) continue;
        double lam = L.lp->primal(cid);
        if (lam > 1e-9) cands.emplace_back(lam, pid);
    }
    std::sort(cands.begin(), cands.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    Solution sol;
    sol.set.assign(m, 0);
    std::vector<char> used(m, 0);
    u64 taken = 0;
    for (const auto& c : cands) {
        const PoolCol& pc = pool_[c.second];
        if (used[pc.v]) continue;
        u64 mask = pc.mask & ~taken;
        if (mask == 0) continue;
        used[pc.v] = 1;
        sol.set[pc.v] = mask;
        taken |= mask;
    }
    sol.cost = eval_.canonical_cost(sol.set, active_mask_);
    if (!std::isfinite(sol.cost)) return;
    update_incumbent(sol, "rounding");
    eval_.improve(sol, active_mask_, clock_, 0.05 * prm_.time_limit);
    update_incumbent(sol, "rounding+ls");
    for (int v = 0; v < m; ++v) add_pool_column(v, sol.set[v]);
}

Result BranchAndPrice::solve() {
    const int n = P_.n, m = P_.m;
    // Big-M for artificial columns: any real column costs less than this.
    {
        double scale = 1.0;
        for (double c : P_.c_out) scale += std::fabs(c);
        for (const Vehicle& vh : P_.veh) {
            double worst = std::fabs(vh.empty_theta);
            for (const Cut& c : vh.cuts) {
                double s = std::fabs(c.a);
                for (double p : c.p) s += std::fabs(p);
                worst = std::max(worst, s);
            }
            scale += worst;
        }
        big_m_ = 4.0 * scale + 1.0;
    }
    hint_heuristic();

    std::priority_queue<BBNode> open;
    BBNode root;
    root.lb = -kInf;
    root.depth = 0;
    root.id = 0;
    root.fix.assign((size_t)n * m, 0);
    root.yfix.assign(m, 0);
    open.push(root);
    long next_id = 1;
    NodeLP L;
    std::vector<double> alpha, yv;
    double global_lb = -kInf;
    std::string status = "optimal";

    auto current_lb = [&](double extra) {
        double lb = std::min(pruned_lb_min_, extra);
        if (!open.empty()) lb = std::min(lb, open.top().lb);
        if (best_.valid) lb = std::min(lb, best_.cost);
        return lb;
    };

    while (!open.empty()) {
        if (clock_.expired()) { status = "time_limit"; break; }
        if (res_.nodes >= prm_.max_nodes) { status = "node_limit"; break; }
        BBNode node = open.top();
        open.pop();
        // plunge: process node, push children, immediately continue with one child
        while (true) {
            if (clock_.expired()) { status = "time_limit"; open.push(node); break; }
            if (res_.nodes >= prm_.max_nodes) { status = "node_limit"; open.push(node); break; }
            if (best_.valid && node.lb >= best_.cost - stop_gap()) {
                if (node.lb < best_.cost) pruned_lb_min_ = std::min(pruned_lb_min_, node.lb);
                break;
            }
            ++res_.nodes;
            bool infeasible = false;
            double lb = solve_node(node, L, infeasible);
            if (!res_.error.empty()) { status = "error"; break; }
            if (infeasible) { if (prm_.verbose >= 2) log("node %ld infeasible", node.id); break; }
            if (lb < node.lb) lb = node.lb;   // a child's bound never drops below the parent's
            if (res_.nodes == 1) res_.root_lb = lb;
            if (prm_.verbose >= 1 && (res_.nodes == 1 || res_.nodes % 50 == 0))
                log("nodes=%ld open=%zu lb=%.6f ub=%.6f cols=%zu cg=%ld pricing_nodes=%ld",
                    res_.nodes, open.size(), current_lb(lb), best_.valid ? best_.cost : kInf,
                    pool_.size(), res_.cg_iterations, res_.pricing_nodes);
            if (clock_.expired()) { status = "time_limit"; node.lb = lb; open.push(node); break; }
            if (best_.valid && lb >= best_.cost - stop_gap()) {
                if (lb < best_.cost) pruned_lb_min_ = std::min(pruned_lb_min_, lb);
                break;
            }
            lp_fractional(L, alpha, yv);
            if (res_.nodes == 1 || (res_.nodes % 8) == 0) rounding_heuristic(L);
            if (best_.valid && lb >= best_.cost - stop_gap()) {
                if (lb < best_.cost) pruned_lb_min_ = std::min(pruned_lb_min_, lb);
                break;
            }
            // branching candidate
            int bv = -1;
            double bfrac = 0.0;
            for (int v = 0; v < m; ++v) {
                double f = std::min(yv[v], 1.0 - yv[v]);
                if (f > 1e-6 && f > bfrac) { bfrac = f; bv = v; }
            }
            int bj = -1, bjv = -1;
            double afrac = 0.0;
            if (bv < 0) {
                for (int j = 0; j < n; ++j) {
                    for (int v = 0; v < m; ++v) {
                        double a = alpha[(size_t)j * m + v];
                        double f = std::min(a, 1.0 - a);
                        if (f > 1e-6 && (f > afrac + 1e-12 ||
                                         (std::fabs(f - afrac) <= 1e-12 && bj >= 0 &&
                                          P_.c_out[j] > P_.c_out[bj]))) {
                            afrac = f; bj = j; bjv = v;
                        }
                    }
                }
            }
            if (bv < 0 && bj < 0) {
                // integral LP solution: extract it
                Solution sol;
                sol.set.assign(m, 0);
                for (int j = 0; j < n; ++j)
                    for (int v = 0; v < m; ++v)
                        if (alpha[(size_t)j * m + v] > 0.5) sol.set[v] |= 1ULL << j;
                std::vector<u64> copy = sol.set;
                sol.cost = eval_.canonical_cost(copy, active_mask_);
                if (copy == sol.set) update_incumbent(sol, "lp-integral");
                else {
                    // LP-integral but not canonical up to ties: keep the canonical
                    // version as an incumbent candidate; node bound lb still valid
                    Solution c2; c2.set = copy; c2.cost = sol.cost; update_incumbent(c2, "lp-integral-canon");
                }
                if (lb < best_.cost) pruned_lb_min_ = std::min(pruned_lb_min_, lb);
                break;
            }
            if (prm_.verbose >= 2)
                log("node %ld depth=%d lb=%.6f ub=%.6f open=%zu cols=%zu branch=%s",
                    node.id, node.depth, lb, best_.valid ? best_.cost : kInf, open.size(),
                    pool_.size(), bv >= 0 ? "y" : "alpha");
            BBNode c1 = node, c2 = node;
            c1.lb = c2.lb = lb;
            c1.depth = c2.depth = node.depth + 1;
            c1.id = next_id++;
            c2.id = next_id++;
            if (bv >= 0) {
                c1.yfix[bv] = 1;    // used
                c2.yfix[bv] = -1;   // unused
            } else {
                c1.fix[(size_t)bj * m + bjv] = 1;
                for (int v = 0; v < m; ++v) if (v != bjv) c1.fix[(size_t)bj * m + v] = -1;
                c2.fix[(size_t)bj * m + bjv] = -1;
            }
            open.push(c2);
            node = std::move(c1);
        }
        if (status != "optimal") break;
        // bound_stop: stop as soon as the global bound proves the piece is not the minimiser
        if (std::isfinite(prm_.bound_stop) && current_lb(kInf) >= prm_.bound_stop) {
            status = "bound_stop";
            break;
        }
        if (best_.valid && !open.empty() && open.top().lb >= best_.cost - stop_gap()) {
            // everything remaining is within the gap
            while (!open.empty()) {
                if (open.top().lb < best_.cost) pruned_lb_min_ = std::min(pruned_lb_min_, open.top().lb);
                open.pop();
            }
        }
    }
    global_lb = current_lb(kInf);
    if (status == "optimal" && open.empty()) {
        if (!best_.valid) { status = "infeasible"; global_lb = kInf; }
    }
    Result r = res_;
    r.seconds = clock_.elapsed();
    r.columns = (long)pool_.size();
    r.has_solution = best_.valid;
    r.status = status;
    if (best_.valid) {
        r.ub = best_.cost;
        r.assign.assign(n, -1);
        for (int v = 0; v < m; ++v)
            for (u64 mm = best_.set[v]; mm; mm &= mm - 1) r.assign[__builtin_ctzll(mm)] = v;
    }
    // Node bounds already carry their floating-point allowances.  The bound is
    // reported unclamped: the caller re-derives the incumbent cost exactly and
    // clamps against that value, not against the binary64 ``ub`` here.
    r.lb = global_lb;
    r.optimal = best_.valid && status == "optimal";
    delete L.lp;
    return r;
}

}  // namespace s2bp

// --------------------------------------------------------------------------
// pybind11 interface
// --------------------------------------------------------------------------

static py::dict solve_py(
    const std::vector<double>& c_out,
    const std::vector<long long>& q,
    const std::vector<long long>& w,
    const std::vector<long long>& caps,
    const std::vector<int>& types,
    const std::vector<double>& empty_theta,
    const std::vector<std::vector<double>>& cut_a,
    const std::vector<std::vector<std::vector<double>>>& cut_p,
    const std::vector<int>& hint,
    const py::dict& params) {
    using namespace s2bp;
    Problem P;
    P.n = (int)c_out.size();
    P.m = (int)caps.size();
    if (P.n > kMaxCustomers) throw std::invalid_argument("s2_bp_kernel supports at most 64 active customers");
    if ((int)q.size() != P.n || (int)w.size() != P.n) throw std::invalid_argument("q/w shape mismatch");
    if ((int)types.size() != P.m || (int)empty_theta.size() != P.m ||
        (int)cut_a.size() != P.m || (int)cut_p.size() != P.m)
        throw std::invalid_argument("vehicle arrays shape mismatch");
    P.c_out = c_out;
    P.q.assign(q.begin(), q.end());
    P.w.assign(w.begin(), w.end());
    for (int j = 0; j < P.n; ++j) {
        if (P.q[j] < 0 || P.w[j] < 0) throw std::invalid_argument("volumes/scores must be non-negative");
        if (!std::isfinite(P.c_out[j])) throw std::invalid_argument("non-finite outsourcing cost");
    }
    P.veh.resize(P.m);
    for (int v = 0; v < P.m; ++v) {
        Vehicle& vh = P.veh[v];
        vh.cap = caps[v];
        vh.type = types[v];
        vh.empty_theta = empty_theta[v];
        if (!std::isfinite(vh.empty_theta)) throw std::invalid_argument("non-finite empty theta");
        if (cut_a[v].size() != cut_p[v].size()) throw std::invalid_argument("cut shape mismatch");
        for (size_t r = 0; r < cut_a[v].size(); ++r) {
            if ((int)cut_p[v][r].size() != P.n) throw std::invalid_argument("cut coefficient shape mismatch");
            Cut c;
            c.a = cut_a[v][r];
            c.p = cut_p[v][r];
            if (!std::isfinite(c.a)) throw std::invalid_argument("non-finite cut intercept");
            for (double x : c.p) if (!std::isfinite(x)) throw std::invalid_argument("non-finite cut coefficient");
            vh.cuts.push_back(std::move(c));
        }
        if (v > 0 && types[v] < types[v - 1]) throw std::invalid_argument("vehicles must be grouped by type in canonical order");
    }
    P.finalize();

    Params prm;
    auto getf = [&](const char* key, double dflt) {
        return params.contains(key) ? params[key].cast<double>() : dflt;
    };
    auto geti = [&](const char* key, long dflt) {
        return params.contains(key) ? params[key].cast<long>() : dflt;
    };
    prm.time_limit = getf("time_limit", prm.time_limit);
    prm.rel_gap = getf("rel_gap", prm.rel_gap);
    prm.abs_gap = getf("abs_gap", prm.abs_gap);
    prm.bound_stop = getf("bound_stop", prm.bound_stop);
    prm.max_nodes = geti("max_nodes", prm.max_nodes);
    prm.pricing_node_limit = geti("pricing_node_limit", prm.pricing_node_limit);
    prm.max_cols_per_pricing = (int)geti("max_cols_per_pricing", prm.max_cols_per_pricing);
    prm.simplex_iter_limit = geti("simplex_iter_limit", prm.simplex_iter_limit);
    prm.verbose = (int)geti("verbose", prm.verbose);
    prm.seed = (unsigned)geti("seed", (long)prm.seed);
    if (!(prm.time_limit > 0.0)) prm.time_limit = 1e-3;

    Result r;
    {
        py::gil_scoped_release release;
        BranchAndPrice bp(P, prm, hint);
        r = bp.solve();
    }
    py::dict out;
    out["assign"] = r.assign;
    out["ub"] = r.ub;
    out["lb"] = r.lb;
    out["optimal"] = r.optimal;
    out["has_solution"] = r.has_solution;
    out["status"] = r.status;
    out["nodes"] = r.nodes;
    out["cg_iterations"] = r.cg_iterations;
    out["pricing_nodes"] = r.pricing_nodes;
    out["columns"] = r.columns;
    out["root_lb"] = r.root_lb;
    out["seconds"] = r.seconds;
    out["error"] = r.error;
    return out;
}

PYBIND11_MODULE(s2_bp_kernel, mod) {
    mod.doc() = "Branch-and-price kernel for the fixed-fleet Stage-2 assignment surrogate";
    mod.def("solve", &solve_py,
            py::arg("c_out"), py::arg("q"), py::arg("w"), py::arg("caps"), py::arg("types"),
            py::arg("empty_theta"), py::arg("cut_a"), py::arg("cut_p"), py::arg("hint"),
            py::arg("params"));
    mod.attr("MAX_CUSTOMERS") = s2bp::kMaxCustomers;
}
