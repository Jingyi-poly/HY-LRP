// ============================================================================
// espprc.cpp -- forward-labeling Elementary Shortest Path Problem with
//                Resource Constraints (ESPPRC) for single-vehicle PCTSP.
//
// 给定:
//   N 个节点 (前 n_active = N-2 个是 active customer, idx n_active = depot_start,
//     idx n_active+1 = depot_end),
//   弧成本 cost[i][j],
//   节点 prize pi_alpha[i] (depot 取 0),
//   节点资源消耗 vol[i] (depot 取 0),
//   capacity (车容量),
//   cutoff (剪枝阈值, path_cost >= cutoff 的 label 全丢),
//   top_k (返回最优 K 条 path).
//
// 求: 从 depot_start 到 depot_end 的 elementary 路径 P, 满足
//   sum_{j in P\{depots}} vol[j] <= capacity,
// 最小化 path_cost = sum_{(i,j) in P} (cost[i,j] - pi_alpha[j]).
//
// 算法: forward labeling + dominance.
//   Label = (node, visited_bitmask, cap_used, cost, parent_id).
//   Dominance: L1 weak-dominates L2 在同 node 上 iff
//     L1.visited subset_of L2.visited
//     L1.cap_used <= L2.cap_used
//     L1.cost     <= L2.cost
//
// LSBC-strict: full elementary, 不用 ng-route relaxation. 状态空间最差 2^n
// 但实际靠 capacity + dominance + cutoff 剪到很小.
//
// 限制: 最多 256 个 active customer (four-word visited bitset).
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iterator>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();
constexpr int MAX_ACTIVE_BITS = 256;  // supports large active sets (e.g., 200)
constexpr int MAX_VISITED_WORDS = MAX_ACTIVE_BITS / 64;
// dominance 比较里使用严格 <=, 不带容忍度 (容忍度方向错了会丢最优).
// Resource feasibility is strict in the binary64 input domain.  Adding an
// empirical epsilon here can return an over-capacity incumbent and then
// incorrectly label it exact/certified.
constexpr double EPS_LB_PRUNE = 1e-9;

using VisitedMask = std::array<uint64_t, MAX_VISITED_WORDS>;

struct WallTimer {
    std::chrono::steady_clock::time_point start;
    double limit_s;

    explicit WallTimer(double limit)
        : start(std::chrono::steady_clock::now()), limit_s(limit) {}

    inline bool expired() const {
        if (limit_s <= 0.0) return false;
        return elapsed_s() >= limit_s;
    }

    inline double elapsed_s() const {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
    }
};

inline void clear_visited(VisitedMask& m) {
    m.fill(0ULL);
}

inline bool is_visited(const VisitedMask& m, int idx) {
    const int w = idx >> 6;
    const int b = idx & 63;
    return ((m[w] >> b) & 1ULL) != 0ULL;
}

inline void set_visited(VisitedMask& m, int idx) {
    const int w = idx >> 6;
    const int b = idx & 63;
    m[w] |= (1ULL << b);
}

inline bool subset_of(const VisitedMask& a, const VisitedMask& b, int n_words) {
    for (int w = 0; w < n_words; ++w) {
        if ((a[w] & ~b[w]) != 0ULL) return false;
    }
    return true;
}

inline bool equals_mask(const VisitedMask& a, const VisitedMask& b, int n_words) {
    for (int w = 0; w < n_words; ++w) {
        if (a[w] != b[w]) return false;
    }
    return true;
}

// Exact single-word operations for the common Taillard/Compare regime.  The
// overloads deliberately accept the same n_words argument as the 256-bit
// implementation so solve_pctsp_core can share one compiled algorithm body.
inline void clear_visited(uint64_t& m) {
    m = 0ULL;
}

inline bool is_visited(uint64_t m, int idx) {
    return ((m >> idx) & 1ULL) != 0ULL;
}

inline void set_visited(uint64_t& m, int idx) {
    m |= (1ULL << idx);
}

inline bool subset_of(uint64_t a, uint64_t b, int /*n_words*/) {
    return (a & ~b) == 0ULL;
}

inline bool equals_mask(uint64_t a, uint64_t b, int /*n_words*/) {
    return a == b;
}

struct Label {
    uint32_t node;        // 节点 index 0..N-1
    uint32_t parent;      // 前一个 label 的 id (UINT32_MAX = root)
    VisitedMask visited;  // active-customer visited bitset (depot not tracked)
    double   cap_used;    // 累计 volume
    double   cost;        // 累计 reduced cost
    bool     alive;       // 被 dominate 后置 false (lazy 删)
};

struct Result {
    std::vector<std::pair<double, std::vector<int>>> paths;  // (cost, node-seq)
    int status;            // 0=ok, 1=infeas (depot_end 没 label), 2=cutoff
    uint64_t n_labels;     // 总创建 label 数 (诊断)
    uint64_t n_pruned_dom; // 被 dominance 剪掉数 (诊断)
    uint64_t n_pruned_lb;  // 被 completion-LB 剪掉数 (诊断)
};

// ============================================================================
// PctspResult -- PCTSP Lagrangian 子问题的完整解
// ============================================================================
//
// lb 字段语义 (LB on optimal full_obj):
//   - status==0 (exact OPT)     ⇒ lb <= OPT <= obj_val; the two endpoints
//       are directed binary64 roundings and may differ by one ulp.
//   - status==2 (time/budget)   ⇒ max(上一轮完整 ng 松弛的最优值, 当前 frontier
//       最小 key), 是合法 LB on Q(π);
//   - status==1 (infeas)        ⇒ lb = +INF (vacuous: 没有 feasible candidate)
//
// caller uses the certified lower endpoint for cuts and the feasible upper
// endpoint for incumbents.
// Optional primal witnesses only. These never participate in search, pruning,
// incumbent selection, status or lower-bound certification.
struct PctspPrimalRoute {
    double obj_val = 0.0;
    int y = 1;
    std::vector<int> path;
    std::vector<int> alpha_inactive;
};

struct PctspResult {
    double obj_val;
    double lb = INF;                  // LB on the optimal full_obj (见上)
    int    y;                         // 0 or 1 (vehicle used?)
    std::vector<int> path;            // route node seq (empty if y=0)
    std::vector<int> alpha_inactive;  // 0/1 per inactive customer
    std::vector<PctspPrimalRoute> candidate_routes;  // populated only for top_k>1
    int    status;                    // 0=ok,1=infeas,2=time/budget interruption
    uint64_t n_labels;
    uint64_t n_pruned_dom;
    uint64_t n_pruned_lb;
    bool timed_out = false;
    bool label_budget_exhausted = false;
    bool cutoff_bound_triggered = false;
    // Timings are measured inside C++, so Python/process scheduling noise is
    // not mixed into the ESP profile.  For the certifying unidirectional
    // solver, ``preprocess_time_s`` covers all work before the label loop and
    // ``label_time_s`` covers the loop plus interrupted-frontier LB recovery.
    double preprocess_time_s = 0.0;
    double label_time_s = 0.0;
    double total_time_s = 0.0;
    // Certifying-UNI profile.  Counters describe exact work performed and do
    // not participate in pruning or bound certification.
    double completion_lb_time_s = 0.0;
    double knapsack_setup_time_s = 0.0;
    double arc_elimination_time_s = 0.0;
    double greedy_time_s = 0.0;
    double frontier_lb_time_s = 0.0;
    double postprocess_time_s = 0.0;
    uint64_t n_labels_popped = 0;
    uint64_t n_arcs_considered = 0;
    uint64_t n_pruned_arc = 0;
    uint64_t n_pruned_elementary = 0;
    uint64_t n_pruned_capacity = 0;
    uint64_t n_dom_checks_forward = 0;
    uint64_t n_dom_checks_reverse = 0;
    uint64_t max_bucket_size = 0;
    uint64_t n_final_labels = 0;
    uint64_t n_pending_labels = 0;
    // ng-route/DSSR certifying core profile.
    int dssr_iterations = 0;
    int n_customers_dropped = 0;
    int ng_size_used = 0;
    int bound_ng_size = 0;         // ng size of the final completion-bound DP
    int bound_level = 0;           // precision level of the final table
    int n_escalations = 0;         // relaxations aborted to build a stronger table
    double root_lb = -INF;         // completion-bound value at the root
    double relaxation_lb = -INF;   // best completed-iteration relaxed optimum
};

// ============================================================================
// 0-1 Knapsack (branch-and-bound with fractional UB)
// ============================================================================
struct KnapsackItem {
    int    idx;
    double profit;
    double weight;
    double ratio;
};

struct KnapsackPrep {
    std::vector<KnapsackItem> items;  // profit>0, weight>0, sorted by ratio desc
    std::vector<int> free_wins;       // profit>0, weight<=0 => always take
    double free_profit;
    double free_profit_ub;
};

// Exact comparisons of positive binary64 products.  Each finite positive
// double is an integer significand (at most 53 bits) times a power of two, so
// a product fits in 106 bits and can be ordered losslessly in uint128.  This
// avoids rounded p/w ties putting a lower-density knapsack item first.
struct PositiveBinary64 {
    uint64_t significand;
    int exponent;
};

inline PositiveBinary64 decompose_positive_binary64(double value) {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint64_t fraction = bits & ((1ULL << 52) - 1ULL);
    const int biased_exp = (int)((bits >> 52) & 0x7ffU);
    if (biased_exp == 0) {
        return {fraction, -1074};
    }
    return {(1ULL << 52) | fraction, biased_exp - 1023 - 52};
}

inline int uint128_bit_width(__uint128_t value) {
    const uint64_t hi = (uint64_t)(value >> 64);
    if (hi != 0) return 128 - __builtin_clzll(hi);
    const uint64_t lo = (uint64_t)value;
    return lo == 0 ? 0 : 64 - __builtin_clzll(lo);
}

inline int compare_scaled_uint128(__uint128_t lhs, int lhs_exp,
                                  __uint128_t rhs, int rhs_exp) {
    if (lhs == 0 || rhs == 0) {
        return lhs == rhs ? 0 : (lhs == 0 ? -1 : 1);
    }
    const int lhs_bits = uint128_bit_width(lhs);
    const int rhs_bits = uint128_bit_width(rhs);
    const int lhs_top = lhs_exp + lhs_bits;
    const int rhs_top = rhs_exp + rhs_bits;
    if (lhs_top != rhs_top) return lhs_top < rhs_top ? -1 : 1;
    if (lhs_exp > rhs_exp) lhs <<= (lhs_exp - rhs_exp);
    if (rhs_exp > lhs_exp) rhs <<= (rhs_exp - lhs_exp);
    return lhs == rhs ? 0 : (lhs < rhs ? -1 : 1);
}

inline int compare_positive_products(double a, double b,
                                     double c, double d) {
    const PositiveBinary64 pa = decompose_positive_binary64(a);
    const PositiveBinary64 pb = decompose_positive_binary64(b);
    const PositiveBinary64 pc = decompose_positive_binary64(c);
    const PositiveBinary64 pd = decompose_positive_binary64(d);
    return compare_scaled_uint128(
        (__uint128_t)pa.significand * pb.significand,
        pa.exponent + pb.exponent,
        (__uint128_t)pc.significand * pd.significand,
        pc.exponent + pd.exponent);
}

inline bool exact_density_greater(const KnapsackItem& a,
                                  const KnapsackItem& b) {
    if (std::isfinite(a.profit) && std::isfinite(a.weight)
        && std::isfinite(b.profit) && std::isfinite(b.weight)) {
        return compare_positive_products(
            a.profit, b.weight, b.profit, a.weight) > 0;
    }
    return a.ratio > b.ratio;
}

// Directed-up elementary operations for certificate-only upper bounds.  The
// exact integer Taillard operations remain bit-identical: an ulp is added only
// when an error-free residual proves round-to-nearest went downward.
inline double add_nonnegative_up(double a, double b) {
    const double rounded = a + b;
    if (!std::isfinite(rounded)) return rounded;
    const double b_virtual = rounded - a;
    const double error = (a - (rounded - b_virtual)) + (b - b_virtual);
    return error > 0.0 ? std::nextafter(rounded, INF) : rounded;
}

inline double add_binary64_down(double a, double b) {
    const double rounded = a + b;
    if (std::isinf(rounded)) {
        if (rounded > 0.0 && std::isfinite(a) && std::isfinite(b)) {
            return std::numeric_limits<double>::max();
        }
        return rounded;
    }
    if (std::isnan(rounded)) return -INF;
    const double b_virtual = rounded - a;
    const double error = (a - (rounded - b_virtual)) + (b - b_virtual);
    return error < 0.0 ? std::nextafter(rounded, -INF) : rounded;
}

inline double add_binary64_up(double a, double b) {
    const double rounded = a + b;
    if (std::isinf(rounded)) {
        if (rounded < 0.0 && std::isfinite(a) && std::isfinite(b)) {
            return -std::numeric_limits<double>::max();
        }
        return rounded;
    }
    if (std::isnan(rounded)) return INF;
    const double b_virtual = rounded - a;
    const double error = (a - (rounded - b_virtual)) + (b - b_virtual);
    return error > 0.0 ? std::nextafter(rounded, INF) : rounded;
}

inline double subtract_binary64_down(double a, double b) {
    return add_binary64_down(a, -b);
}

inline double subtract_binary64_up(double a, double b) {
    return add_binary64_up(a, -b);
}

// Every finite binary64 value is an integer multiple of 2^-1074.  A fixed
// 2560-bit signed accumulator therefore represents any sum occurring in this
// <=256-customer backend exactly (including full exponent-range cancellation).
// It is used only in cold candidate-order/reporting paths; search order remains
// the historical binary64 order.
struct ExactBinarySum {
    static constexpr size_t WORDS = 40;
    std::array<uint64_t, WORDS> magnitude{};
    bool negative = false;

    bool is_zero() const {
        for (uint64_t word : magnitude) if (word != 0) return false;
        return true;
    }

    int bit_count() const {
        for (size_t pos = WORDS; pos-- > 0;) {
            if (magnitude[pos] != 0) {
                return (int)(pos * 64U + 64U
                    - (unsigned)__builtin_clzll(magnitude[pos]));
            }
        }
        return 0;
    }

    static int compare_magnitude(const ExactBinarySum& a,
                                 const ExactBinarySum& b) {
        for (size_t pos = WORDS; pos-- > 0;) {
            if (a.magnitude[pos] != b.magnitude[pos]) {
                return a.magnitude[pos] < b.magnitude[pos] ? -1 : 1;
            }
        }
        return 0;
    }

    void normalize_zero() {
        if (is_zero()) negative = false;
    }

    static void add_magnitudes(ExactBinarySum& target,
                               const ExactBinarySum& other) {
        uint64_t carry = 0;
        for (size_t pos = 0; pos < WORDS; ++pos) {
            const uint64_t first = target.magnitude[pos] + other.magnitude[pos];
            const uint64_t carry_first = first < target.magnitude[pos];
            const uint64_t second = first + carry;
            const uint64_t carry_second = second < first;
            target.magnitude[pos] = second;
            carry = carry_first | carry_second;
        }
        if (carry != 0) {
            throw std::overflow_error("exact binary64 accumulator overflow");
        }
    }

    static void subtract_magnitudes(ExactBinarySum& larger,
                                    const ExactBinarySum& smaller) {
        uint64_t borrow = 0;
        for (size_t pos = 0; pos < WORDS; ++pos) {
            const uint64_t first = larger.magnitude[pos] - smaller.magnitude[pos];
            const uint64_t borrow_first = larger.magnitude[pos]
                < smaller.magnitude[pos];
            const uint64_t second = first - borrow;
            const uint64_t borrow_second = first < borrow;
            larger.magnitude[pos] = second;
            borrow = borrow_first | borrow_second;
        }
    }

    void add_signed(const ExactBinarySum& other) {
        if (other.is_zero()) return;
        if (is_zero()) {
            *this = other;
            return;
        }
        if (negative == other.negative) {
            add_magnitudes(*this, other);
            return;
        }
        const int cmp = compare_magnitude(*this, other);
        if (cmp == 0) {
            magnitude.fill(0);
            negative = false;
        } else if (cmp > 0) {
            subtract_magnitudes(*this, other);
        } else {
            ExactBinarySum result = other;
            subtract_magnitudes(result, *this);
            *this = result;
        }
        normalize_zero();
    }

    bool bit_at(unsigned bit) const {
        return ((magnitude[bit >> 6U] >> (bit & 63U)) & 1ULL) != 0;
    }

    bool any_bits_below(unsigned exclusive_bit) const {
        const unsigned full_words = exclusive_bit >> 6U;
        for (unsigned pos = 0; pos < full_words; ++pos) {
            if (magnitude[pos] != 0) return true;
        }
        const unsigned tail = exclusive_bit & 63U;
        if (tail != 0U) {
            const uint64_t mask = (1ULL << tail) - 1ULL;
            if ((magnitude[full_words] & mask) != 0) return true;
        }
        return false;
    }

    uint64_t shifted_low_word(unsigned shift) const {
        const unsigned word = shift >> 6U;
        const unsigned bit = shift & 63U;
        uint64_t result = magnitude[word] >> bit;
        if (bit != 0U && word + 1U < WORDS) {
            result |= magnitude[word + 1U] << (64U - bit);
        }
        return result;
    }
};

inline bool operator==(const ExactBinarySum& a, const ExactBinarySum& b) {
    return a.negative == b.negative && a.magnitude == b.magnitude;
}
inline bool operator!=(const ExactBinarySum& a, const ExactBinarySum& b) {
    return !(a == b);
}
inline bool operator<(const ExactBinarySum& a, const ExactBinarySum& b) {
    if (a.negative != b.negative) return a.negative;
    const int cmp = ExactBinarySum::compare_magnitude(a, b);
    return a.negative ? cmp > 0 : cmp < 0;
}
inline bool operator>(const ExactBinarySum& a, const ExactBinarySum& b) {
    return b < a;
}
inline bool operator<=(const ExactBinarySum& a, const ExactBinarySum& b) {
    return !(b < a);
}
inline bool operator>=(const ExactBinarySum& a, const ExactBinarySum& b) {
    return !(a < b);
}

inline ExactBinarySum binary64_scaled_integer(double value) {
    ExactBinarySum scaled;
    if (value == 0.0) return scaled;
    if (!std::isfinite(value)) {
        throw std::runtime_error("exact accumulator requires finite binary64");
    }
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    scaled.negative = (bits >> 63) != 0;
    const uint64_t fraction = bits & ((1ULL << 52) - 1ULL);
    const int biased_exp = (int)((bits >> 52) & 0x7ffU);
    const uint64_t significand = biased_exp == 0
        ? fraction : ((1ULL << 52) | fraction);
    const unsigned shift = biased_exp == 0 ? 0U
        : (unsigned)(biased_exp - 1);
    const unsigned word = shift >> 6U;
    const unsigned bit = shift & 63U;
    scaled.magnitude[word] |= significand << bit;
    if (bit != 0U) {
        scaled.magnitude[word + 1U] |= significand >> (64U - bit);
    }
    scaled.normalize_zero();
    return scaled;
}

inline void add_binary64_exact(ExactBinarySum& sum, double value) {
    sum.add_signed(binary64_scaled_integer(value));
}

inline void subtract_binary64_exact(ExactBinarySum& sum, double value) {
    ExactBinarySum term = binary64_scaled_integer(value);
    if (!term.is_zero()) term.negative = !term.negative;
    sum.add_signed(term);
}

inline double scaled_integer_to_nearest_binary64(
    const ExactBinarySum& value) {
    const int bit_count = value.bit_count();
    if (bit_count == 0) return 0.0;
    unsigned shift = bit_count > 53 ? (unsigned)(bit_count - 53) : 0U;
    uint64_t significand = value.shifted_low_word(shift);
    if (shift > 0U) {
        const bool half = value.bit_at(shift - 1U);
        const bool lower = value.any_bits_below(shift - 1U);
        if (half && (lower || (significand & 1ULL) != 0ULL)) {
            ++significand;
            if (significand == (1ULL << 53)) {
                significand >>= 1U;
                ++shift;
            }
        }
    }
    const double rounded = std::ldexp(
        (double)significand, (int)shift - 1074);
    return value.negative ? -rounded : rounded;
}

inline double scaled_integer_to_binary64_up(const ExactBinarySum& value) {
    double rounded = scaled_integer_to_nearest_binary64(value);
    if (rounded == INF) return rounded;
    if (rounded == -INF) return -std::numeric_limits<double>::max();
    if (binary64_scaled_integer(rounded) < value) {
        rounded = std::nextafter(rounded, INF);
    }
    return rounded;
}

inline double scaled_integer_to_binary64_down(const ExactBinarySum& value) {
    double rounded = scaled_integer_to_nearest_binary64(value);
    if (rounded == -INF) return rounded;
    if (rounded == INF) return std::numeric_limits<double>::max();
    if (binary64_scaled_integer(rounded) > value) {
        rounded = std::nextafter(rounded, -INF);
    }
    return rounded;
}

inline ExactBinarySum exact_path_reduced_cost(
    const std::vector<int>& path,
    const double* cost, const double* pi_alpha, int N) {
    ExactBinarySum exact;
    for (size_t pos = 1; pos < path.size(); ++pos) {
        const int i = path[pos - 1];
        const int j = path[pos];
        add_binary64_exact(exact, cost[(size_t)i * N + j]);
        subtract_binary64_exact(exact, pi_alpha[j]);
    }
    return exact;
}

inline ExactBinarySum exact_pctsp_objective(
    int y,
    const std::vector<int>& path,
    const std::vector<int>& inactive_selected,
    const double* cost, const double* pi_alpha, int N,
    const double* pi_alpha_inactive, double pi_y) {
    if (y == 0) return ExactBinarySum{};
    ExactBinarySum exact = exact_path_reduced_cost(path, cost, pi_alpha, N);
    for (int idx : inactive_selected) {
        subtract_binary64_exact(exact, pi_alpha_inactive[idx]);
    }
    subtract_binary64_exact(exact, pi_y);
    return exact;
}

template <typename AnyLabel>
inline ExactBinarySum exact_label_reduced_cost(
    const std::vector<AnyLabel>& labels, uint32_t lid,
    const double* cost, const double* pi_alpha, int N) {
    ExactBinarySum exact;
    uint32_t child = lid;
    while (labels[child].parent != UINT32_MAX) {
        const uint32_t parent = labels[child].parent;
        const int i = (int)labels[parent].node;
        const int j = (int)labels[child].node;
        add_binary64_exact(exact, cost[(size_t)i * N + j]);
        subtract_binary64_exact(exact, pi_alpha[j]);
        child = parent;
    }
    return exact;
}

template <typename AnyLabel>
inline ExactBinarySum exact_label_active_volume(
    const std::vector<AnyLabel>& labels, uint32_t lid,
    const double* vol, int n_active) {
    ExactBinarySum exact;
    uint32_t child = lid;
    while (labels[child].parent != UINT32_MAX) {
        const int node = (int)labels[child].node;
        if (node < n_active) add_binary64_exact(exact, vol[node]);
        child = labels[child].parent;
    }
    return exact;
}

inline ExactBinarySum exact_remaining_capacity(
    double capacity, const ExactBinarySum& used) {
    ExactBinarySum remaining = binary64_scaled_integer(capacity);
    ExactBinarySum neg_used = used;
    if (!neg_used.is_zero()) neg_used.negative = !neg_used.negative;
    remaining.add_signed(neg_used);
    return remaining;
}

inline ExactBinarySum exact_path_active_volume(
    const std::vector<int>& path, const double* vol, int n_active) {
    ExactBinarySum used;
    for (int node : path) {
        if (node >= 0 && node < n_active) {
            add_binary64_exact(used, vol[node]);
        }
    }
    return used;
}

inline double pctsp_full_lower_bound(double route_lb,
                                     double inactive_profit_ub,
                                     double pi_y) {
    return subtract_binary64_down(
        subtract_binary64_down(route_lb, inactive_profit_ub), pi_y);
}

inline double subtract_nonnegative_up(double a, double b) {
    const double rounded = a - b;
    if (!std::isfinite(rounded)) return rounded;
    const double neg_b = -b;
    const double b_virtual = rounded - a;
    const double error = (a - (rounded - b_virtual)) + (neg_b - b_virtual);
    return error > 0.0 ? std::nextafter(rounded, INF) : rounded;
}

inline double remaining_capacity_up(double capacity, double used) {
    if (used >= capacity) return 0.0;
    return std::max(0.0, subtract_nonnegative_up(capacity, used));
}

inline double multiply_nonnegative_up(double a, double b) {
    const double rounded = a * b;
    if (std::isinf(rounded) || a == 0.0 || b == 0.0) return rounded;
    if (rounded == 0.0) return std::nextafter(0.0, INF);
    const PositiveBinary64 pa = decompose_positive_binary64(a);
    const PositiveBinary64 pb = decompose_positive_binary64(b);
    const PositiveBinary64 pr = decompose_positive_binary64(rounded);
    const int cmp = compare_scaled_uint128(
        (__uint128_t)pa.significand * pb.significand,
        pa.exponent + pb.exponent,
        pr.significand, pr.exponent);
    return cmp > 0 ? std::nextafter(rounded, INF) : rounded;
}

inline double divide_positive_up(double numerator, double denominator) {
    const double rounded = numerator / denominator;
    if (std::isinf(rounded)) return rounded;
    if (rounded == 0.0) return std::nextafter(0.0, INF);
    const PositiveBinary64 pq = decompose_positive_binary64(rounded);
    const PositiveBinary64 pd = decompose_positive_binary64(denominator);
    const PositiveBinary64 pn = decompose_positive_binary64(numerator);
    const int cmp = compare_scaled_uint128(
        (__uint128_t)pq.significand * pd.significand,
        pq.exponent + pd.exponent,
        pn.significand, pn.exponent);
    return cmp < 0 ? std::nextafter(rounded, INF) : rounded;
}

inline KnapsackPrep prepare_knapsack(const double* profits,
                                     const double* weights,
                                     int n) {
    KnapsackPrep kp;
    kp.free_profit = 0.0;
    kp.free_profit_ub = 0.0;
    for (int i = 0; i < n; ++i) {
        if (profits[i] <= 0.0) continue;
        if (weights[i] <= 0.0) {
            kp.free_wins.push_back(i);
            kp.free_profit += profits[i];
            kp.free_profit_ub = add_nonnegative_up(
                kp.free_profit_ub, profits[i]);
        } else {
            kp.items.push_back({i, profits[i], weights[i],
                                divide_positive_up(profits[i], weights[i])});
        }
    }
    std::sort(kp.items.begin(), kp.items.end(),
              [](const KnapsackItem& a, const KnapsackItem& b) {
                  return exact_density_greater(a, b);
              });
    return kp;
}

inline double frac_knapsack_ub(const std::vector<KnapsackItem>& items,
                               int from, double cap) {
    double ub = 0.0;
    double rem = cap;
    for (int i = from; i < (int)items.size(); ++i) {
        if (rem <= 0.0) break;
        if (items[i].weight <= rem) {
            ub = add_nonnegative_up(ub, items[i].profit);
            rem = subtract_nonnegative_up(rem, items[i].weight);
        } else {
            ub = add_nonnegative_up(
                ub, multiply_nonnegative_up(items[i].ratio, rem));
            break;
        }
    }
    return ub;
}

inline void knap_bb_recurse(const std::vector<KnapsackItem>& items,
                            int depth, double cap, double cur_profit,
                            double cur_profit_ub,
                            double& best_profit,
                            ExactBinarySum& cur_profit_exact,
                            ExactBinarySum& best_profit_exact,
                            ExactBinarySum& cur_weight_exact,
                            const ExactBinarySum& capacity_exact,
                            std::vector<bool>& cur_sel,
                            std::vector<bool>& best_sel,
                            const WallTimer* timer = nullptr,
                            bool* timed_out = nullptr) {
    if (timer && timer->expired()) {
        if (timed_out) *timed_out = true;
        return;
    }
    if (depth == (int)items.size()) {
        if (best_profit_exact < cur_profit_exact) {
            best_profit_exact = cur_profit_exact;
            // A downward-rounded feasible value keeps `ub <= best_profit`
            // pruning conservative when the exact sum is not representable.
            best_profit = scaled_integer_to_binary64_down(best_profit_exact);
            best_sel = cur_sel;
        }
        return;
    }
    const double ub = add_nonnegative_up(
        cur_profit_ub, frac_knapsack_ub(items, depth, cap));
    if (ub <= best_profit) return;

    // try take
    add_binary64_exact(cur_weight_exact, items[depth].weight);
    if (cur_weight_exact <= capacity_exact) {
        cur_sel[depth] = true;
        add_binary64_exact(cur_profit_exact, items[depth].profit);
        knap_bb_recurse(items, depth + 1,
                        subtract_nonnegative_up(cap, items[depth].weight),
                        cur_profit + items[depth].profit,
                        add_nonnegative_up(cur_profit_ub,
                                           items[depth].profit),
                        best_profit,
                        cur_profit_exact, best_profit_exact,
                        cur_weight_exact, capacity_exact,
                        cur_sel, best_sel, timer, timed_out);
        subtract_binary64_exact(cur_profit_exact, items[depth].profit);
        cur_sel[depth] = false;
    }
    subtract_binary64_exact(cur_weight_exact, items[depth].weight);
    if (timed_out && *timed_out) return;
    // try skip
    knap_bb_recurse(items, depth + 1, cap, cur_profit, cur_profit_ub,
                    best_profit,
                    cur_profit_exact, best_profit_exact,
                    cur_weight_exact, capacity_exact,
                    cur_sel, best_sel, timer, timed_out);
}

struct KnapsackSol {
    double profit;
    std::vector<int> selected;  // original indices
};

inline KnapsackSol knapsack_solve_exact_capacity(
    const KnapsackPrep& kp, double cap_ub,
    const ExactBinarySum& capacity_exact,
    const WallTimer* timer = nullptr,
    bool* timed_out = nullptr) {
    KnapsackSol sol;
    sol.profit = kp.free_profit;
    sol.selected = kp.free_wins;
    if (kp.items.empty() || capacity_exact.negative
        || capacity_exact.is_zero()) return sol;

    double best_profit = 0.0;
    ExactBinarySum cur_profit_exact;
    ExactBinarySum best_profit_exact;
    ExactBinarySum cur_weight_exact;
    std::vector<bool> cur_sel(kp.items.size(), false);
    std::vector<bool> best_sel(kp.items.size(), false);
    knap_bb_recurse(kp.items, 0, cap_ub, 0.0, 0.0,
                    best_profit,
                    cur_profit_exact, best_profit_exact,
                    cur_weight_exact, capacity_exact,
                    cur_sel, best_sel, timer, timed_out);
    sol.profit += best_profit;
    for (int i = 0; i < (int)kp.items.size(); ++i) {
        if (best_sel[i]) sol.selected.push_back(kp.items[i].idx);
    }
    return sol;
}

inline KnapsackSol knapsack_solve(const KnapsackPrep& kp, double cap,
                                  const WallTimer* timer = nullptr,
                                  bool* timed_out = nullptr) {
    const ExactBinarySum capacity_exact = binary64_scaled_integer(cap);
    return knapsack_solve_exact_capacity(
        kp, cap, capacity_exact, timer, timed_out);
}

inline KnapsackSol knapsack_solve_nonempty(const KnapsackPrep& kp,
                                           const double* profits,
                                           const double* weights,
                                           int n,
                                           double cap,
                                           const WallTimer* timer = nullptr,
                                           bool* timed_out = nullptr) {
    KnapsackSol sol = knapsack_solve(kp, cap, timer, timed_out);
    if (timed_out && *timed_out) return sol;
    if (!sol.selected.empty()) return sol;

    // The y=1/no-route candidate must assign at least one inactive customer.
    // Even a non-positive profit can be optimal once the -pi_y*y term is counted.
    double best_profit = -INF;
    int best_idx = -1;
    for (int i = 0; i < n; ++i) {
        if ((i & 255) == 0 && timer && timer->expired()) {
            if (timed_out) *timed_out = true;
            return sol;
        }
        if (weights[i] <= cap && profits[i] > best_profit) {
            best_profit = profits[i];
            best_idx = i;
        }
    }
    if (best_idx >= 0) {
        sol.profit = best_profit;
        sol.selected = {best_idx};
    }
    return sol;
}

inline double knapsack_ub(const KnapsackPrep& kp, double cap) {
    return add_nonnegative_up(
        kp.free_profit_ub, frac_knapsack_ub(kp.items, 0, cap));
}

// O(log n) fractional knapsack bound: prefix weights rounded down and prefix
// profits rounded up, so "the first i items fit" is decided optimistically and
// the value is an upper bound of the fractional optimum for every capacity.
struct KnapsackEnvelope {
    std::vector<double> weight_down;   // weight_down[i] <= sum of first i weights
    std::vector<double> profit_up;     // profit_up[i]  >= sum of first i profits
    const KnapsackPrep* kp = nullptr;

    explicit KnapsackEnvelope(const KnapsackPrep& prep) : kp(&prep) {
        const size_t n = prep.items.size();
        weight_down.assign(n + 1, 0.0);
        profit_up.assign(n + 1, 0.0);
        for (size_t i = 0; i < n; ++i) {
            weight_down[i + 1] = add_binary64_down(weight_down[i],
                                                   prep.items[i].weight);
            profit_up[i + 1] = add_nonnegative_up(profit_up[i],
                                                  prep.items[i].profit);
        }
    }
    double ub(double cap) const {
        if (!(cap > 0.0)) return kp->free_profit_ub;
        // largest i with weight_down[i] <= cap
        const size_t i = std::upper_bound(weight_down.begin(), weight_down.end(),
                                          cap) - weight_down.begin() - 1;
        double value = profit_up[i];
        if (i < kp->items.size()) {
            const double rem = subtract_nonnegative_up(cap, weight_down[i]);
            if (rem > 0.0) {
                value = add_nonnegative_up(
                    value, multiply_nonnegative_up(kp->items[i].ratio, rem));
            }
        }
        return add_nonnegative_up(kp->free_profit_ub, value);
    }
};

// ============================================================================
// Knapsack DP table (exact, integer-scaled capacities) + memoized solve cache
// ============================================================================
struct KnapsackDPTable {
    std::vector<double> dp;   // dp[r] = max profit achievable with capacity r
    double scale;             // capacity_scaled = round(capacity * scale)
    int max_cap;              // max scaled capacity
    bool valid;               // whether DP table was successfully built

    KnapsackDPTable() : scale(1.0), max_cap(0), valid(false) {}
};

// Try to build an exact DP/cache key domain.  Only already-integral binary64
// weights qualify: a merely near-integral scaled weight can straddle a cache
// bucket's capacity boundary and make a reused incumbent infeasible (or hide a
// better feasible selection).  Integer weights preserve the exact feasible
// set under floor(capacity), and cover the Taillard production domain.
inline KnapsackDPTable build_knapsack_dp(const KnapsackPrep& kp, double capacity,
                                         const WallTimer* timer = nullptr,
                                         bool* timed_out = nullptr) {
    KnapsackDPTable tbl;
    tbl.valid = false;
    if (kp.items.empty()) {
        // There is nothing to cache.  In particular, never cast an arbitrary
        // finite API capacity to int or allocate O(capacity) entries here.
        return tbl;
    }

    if (!std::isfinite(capacity) || capacity < 0.0
        || capacity > 100000.0) return tbl;
    for (const auto& it : kp.items) {
        if (!std::isfinite(it.weight) || it.weight < 0.0
            || it.weight != std::floor(it.weight)
            || it.weight > 100000.0) return tbl;
    }

    tbl.scale = 1.0;
    tbl.max_cap = (int)std::floor(capacity);
    tbl.dp.assign(tbl.max_cap + 1, 0.0);

    // Standard 0-1 knapsack DP
    for (size_t item_pos = 0; item_pos < kp.items.size(); ++item_pos) {
        if (timer && timer->expired()) {
            if (timed_out) *timed_out = true;
            return tbl;
        }
        const auto& it = kp.items[item_pos];
        int w = (int)it.weight;
        if (w <= 0) continue;
        for (int r = tbl.max_cap; r >= w; --r) {
            if ((r & 4095) == 0 && timer && timer->expired()) {
                if (timed_out) *timed_out = true;
                return tbl;
            }
            double val = tbl.dp[r - w] + it.profit;
            if (val > tbl.dp[r]) tbl.dp[r] = val;
        }
    }
    // Add free_profit
    for (int r = 0; r <= tbl.max_cap; ++r) {
        tbl.dp[r] += kp.free_profit;
    }
    tbl.valid = true;
    return tbl;
}

// Cached knapsack_solve: memoize by scaled remaining capacity
struct KnapsackCache {
    std::vector<KnapsackSol> cache;  // indexed by scaled capacity
    double scale;
    int max_cap;
    bool has_dp;

    KnapsackCache() : scale(1.0), max_cap(0), has_dp(false) {}
};

inline KnapsackCache build_knapsack_cache(const KnapsackPrep& /*kp*/, double capacity,
                                          double dp_scale) {
    KnapsackCache kc;
    kc.scale = dp_scale;
    kc.max_cap = std::max(0, (int)std::floor(capacity * dp_scale));
    kc.has_dp = false;
    // Pre-size but don't fill (lazy)
    kc.cache.resize(kc.max_cap + 1);
    // Mark uncomputed with profit = -INF sentinel
    for (auto& s : kc.cache) s.profit = -INF;
    return kc;
}

inline KnapsackSol& knapsack_solve_cached(KnapsackCache& kc,
                                           const KnapsackPrep& kp,
                                           double cap) {
    int r = (int)(cap * kc.scale);
    if (r > kc.max_cap) r = kc.max_cap;
    if (r < 0) r = 0;
    if (kc.cache[r].profit < -1e300) {
        kc.cache[r] = knapsack_solve(kp, cap);
    }
    return kc.cache[r];
}

// ============================================================================
// === BIDIRECTIONAL LABELING (Righini-Salani 2008 style) ===
//
// 标准 ESPPRC bidirectional acceleration. 关键改动:
//   - 前向 (forward) 从 depot_start 出发, 限制 q_F <= q_split.
//   - 后向 (backward) 从 depot_end 出发反推, 限制 q_B <= q_split.
//   - 在每个 node u 上 join: F@u + arc(u,v) + B@v.
//   - 必须 max(active vol) <= q_split (= cap/2) 才能保证 join completeness;
//     否则回退到单向 (上层判断).
//
// 数学约定 (统一约定 v2, 不对称 prize):
//   F-label at u: 路径 ds -> ... -> u.
//       V_F = active customers visited (含 u if u active)
//       q_F = sum vol[active customers visited] (含 vol[u])
//       C_F = sum_{(a,b) in path} (c[a,b] - pi_alpha[b])  [含 pi_alpha[u]]
//   B-label at h: 路径 h -> ... -> de.
//       V_B = active customers visited in suffix (含 h if h active)
//       q_B = sum vol[active customers in V_B] (含 vol[h])
//       C_B = sum_{(a,b) in suffix path} (c[a,b] - pi_alpha[b])
//             [NOT 含 pi_alpha[h]; h 是 suffix 起点, 它的 prize 由 join arc 补]
//
// Join F@u + arc(u,v) + B@v:
//   完整 path: ds -> ... -> u -> v -> ... -> de
//   q_total = q_F + q_B
//   cost_total = C_F + (c[u,v] - pi_alpha[v]) + C_B
//   必要条件: V_F ∩ V_B = {}  (elementary; u 在 V_F, v 在 V_B, 不冲突)
//             q_F + q_B <= capacity
//
// 反向 dominance 与正向一样形式: same node, V ⊆, q <=, C <=.
// ============================================================================
enum Direction { DIR_FORWARD = 0, DIR_BACKWARD = 1 };

struct LabelStore {
    std::vector<Label> labels;
    std::vector<double> cost_lb;
    std::vector<double> cost_ub;
    std::vector<double> cap_lb;
    std::vector<double> cap_ub;
    std::vector<std::vector<uint32_t>> per_node;  // alive label ids per head node
    uint64_t n_pruned_dom = 0;
    uint64_t n_pruned_lb  = 0;
    bool budget_triggered = false;
    // Bidirectional probes are deliberately non-certifying when interrupted.
    // This flag only says that enumeration stopped at the wall deadline; it
    // must never be interpreted as a valid lower-bound certificate.
    bool timed_out = false;
};

// 计算 completion LB:
//   dir=FORWARD  -> LB[i] = relaxed min cost from i to depot_end.
//   dir=BACKWARD -> LB[i] = relaxed min cost from depot_start to i,
//                   including the prize paid when entering i.
// "Relaxed" = 允许非 elementary (重复访问), 忽略 capacity. Bellman-style 递推.
// A backward label rooted at i excludes i's prize, so adding this prefix LB
// counts every visited customer's prize exactly once.
inline std::vector<double> compute_completion_lb(
    Direction dir,
    const double* cost, const double* pi_alpha,
    int N, int n_active, int depot_start, int depot_end,
    const WallTimer* timer = nullptr,
    bool* timed_out = nullptr)
{
    std::vector<double> LB(N, INF);
    const int max_hops = n_active + 1;
    std::vector<double> prev(N, INF);
    std::vector<double> curr(N, INF);

    if (dir == DIR_FORWARD) {
        // f(i) = min cost from i to depot_end via <= h relaxed arcs.
        prev[depot_end] = 0.0;
        for (int j = 0; j < N; ++j) {
            if (prev[j] < LB[j]) LB[j] = prev[j];
        }
        for (int h = 1; h <= max_hops; ++h) {
            if (timer && timer->expired()) {
                if (timed_out) *timed_out = true;
                return LB;
            }
            for (int j = 0; j < N; ++j) {
                if (j == depot_end) { curr[j] = 0.0; continue; }
                double best = INF;
                for (int m = 0; m < N; ++m) {
                    if ((m & 63) == 0 && timer && timer->expired()) {
                        if (timed_out) *timed_out = true;
                        return LB;
                    }
                    if (m == j) continue;
                    if (m == depot_start) continue;
                    if (prev[m] >= INF) continue;
                    const double arc = subtract_binary64_down(
                        cost[(size_t)j * N + m], pi_alpha[m]);
                    const double tot = add_binary64_down(arc, prev[m]);
                    if (tot < best) best = tot;
                }
                curr[j] = best;
            }
            for (int j = 0; j < N; ++j) {
                if (curr[j] < LB[j]) LB[j] = curr[j];
            }
            std::swap(prev, curr);
        }
    } else {
        // BACKWARD completion: b(i) is a relaxed prefix ds -> ... -> i with
        // the usual reduced arc cost c[m,i]-pi_alpha[i].  It must include the
        // head prize: the backward suffix label at i deliberately excludes
        // that prize, and both backward-label pruning and arc-through bounds
        // add the two quantities.
        prev[depot_start] = 0.0;
        for (int j = 0; j < N; ++j) {
            if (prev[j] < LB[j]) LB[j] = prev[j];
        }
        for (int h = 1; h <= max_hops; ++h) {
            if (timer && timer->expired()) {
                if (timed_out) *timed_out = true;
                return LB;
            }
            for (int j = 0; j < N; ++j) {
                if (j == depot_start) { curr[j] = 0.0; continue; }
                double best = INF;
                for (int m = 0; m < N; ++m) {
                    if ((m & 63) == 0 && timer && timer->expired()) {
                        if (timed_out) *timed_out = true;
                        return LB;
                    }
                    if (m == j) continue;
                    if (m == depot_end) continue;  // de 不能作为中间节点
                    if (prev[m] >= INF) continue;
                    const double arc = subtract_binary64_down(
                        cost[(size_t)m * N + j], pi_alpha[j]);
                    const double tot = add_binary64_down(arc, prev[m]);
                    if (tot < best) best = tot;
                }
                curr[j] = best;
            }
            for (int j = 0; j < N; ++j) {
                if (curr[j] < LB[j]) LB[j] = curr[j];
            }
            std::swap(prev, curr);
        }
    }
    return LB;
}

// 单向 labeling pass (forward 或 backward), q_F/q_B <= q_limit hard cap.
//
// FORWARD: start at depot_start, extend i -> j (j active, or j = depot_end).
//   arc cost = c[i,j] - pi_alpha[j], q_new = q + vol[j] if j active.
// BACKWARD: start at depot_end, "extend i -> j" (j = forward predecessor of i).
//   arc cost = c[j,i] - pi_alpha[i], q_new = q + vol[j] if j active.
//   (j active, or j = depot_start. 但 j = depot_start 也不在 join 候选, 仅作
//   终结标记;  实际不太需要. 这里允许 j = depot_start 以便 backward 走完整 path
//   并 dominate 一些 forward labels at depot_end? 不, 我们 join 不用 ds 那侧.
//   为简单, backward 也允许扩展到 ds, 但 ds 上的 label 不参与 join.)
//
// 注意: 我们在生成新 label 后做 LB-prune: cost + LB[head] > prune_threshold.
//   forward: LB = "i -> depot_end" 的下界.
//   backward: LB = "depot_start -> i" 的下界.
inline LabelStore labeling_pass(
    Direction dir,
    const double* cost, const double* pi_alpha, const double* vol,
    int N, int n_active, int depot_start, int depot_end,
    double q_limit,
    const std::vector<double>& LB_other,
    double prune_threshold,
    uint64_t label_budget,
    const WallTimer* timer = nullptr)
{
    LabelStore S;
    S.per_node.assign(N, std::vector<uint32_t>());
    for (auto& v : S.per_node) v.reserve(64);
    S.labels.reserve(1 << 16);
    S.cost_lb.reserve(1 << 16);
    S.cost_ub.reserve(1 << 16);
    S.cap_lb.reserve(1 << 16);
    S.cap_ub.reserve(1 << 16);

    const int n_words = (n_active + 63) / 64;
    const int source = (dir == DIR_FORWARD) ? depot_start : depot_end;
    const int sink   = (dir == DIR_FORWARD) ? depot_end   : depot_start;

    std::deque<uint32_t> work;
    {
        Label root;
        root.node = (uint32_t)source;
        root.parent = UINT32_MAX;
        clear_visited(root.visited);
        root.cap_used = 0.0;
        root.cost = 0.0;
        root.alive = true;
        S.labels.push_back(root);
        S.cost_lb.push_back(0.0);
        S.cost_ub.push_back(0.0);
        S.cap_lb.push_back(0.0);
        S.cap_ub.push_back(0.0);
        S.per_node[source].push_back(0);
        work.push_back(0);
    }

    while (!work.empty()) {
        if (timer && timer->expired()) {
            S.timed_out = true;
            break;
        }
        uint32_t lid = work.front();
        work.pop_front();
        if (!S.labels[lid].alive) continue;
        Label L = S.labels[lid];

        // sink-reached labels are kept (they will join with the trivial
        // root on the other side, giving full single-direction paths).
        // 不再 extend them (no out arcs from sink in same direction).
        if ((int)L.node == sink) continue;

        // pre-extend LB prune on current label
        double lb_here = LB_other[L.node];
        if (lb_here < INF
            && add_binary64_down(S.cost_lb[lid], lb_here)
                > prune_threshold + EPS_LB_PRUNE) {
            ++S.n_pruned_lb;
            continue;
        }

        // candidate next nodes:
        // FORWARD: j ∈ {active} ∪ {depot_end}, j != L.node, j != depot_start
        // BACKWARD: j ∈ {active} ∪ {depot_start}, j != L.node, j != depot_end
        //   (j is "the new head", i.e. forward predecessor in BACKWARD case)
        for (int j = 0; j < N; ++j) {
            if (timer && timer->expired()) {
                S.timed_out = true;
                break;
            }
            if (j == (int)L.node) continue;
            if (dir == DIR_FORWARD) {
                if (j == depot_start) continue;
            } else {
                if (j == depot_end) continue;
            }
            // elementary: active customer 只能 visit 一次
            if (j < n_active) {
                if (is_visited(L.visited, j)) continue;
            }

            // cap & arc cost
            double new_cap = L.cap_used;
            if (j < n_active) new_cap += vol[j];
            double new_cap_lb = S.cap_lb[lid];
            double new_cap_ub = S.cap_ub[lid];
            if (j < n_active) {
                new_cap_lb = add_binary64_down(new_cap_lb, vol[j]);
                new_cap_ub = add_binary64_up(new_cap_ub, vol[j]);
            }
            if (new_cap_lb > q_limit) continue;

            double arc_c, prize_sub;
            if (dir == DIR_FORWARD) {
                arc_c = cost[(size_t)L.node * N + j];
                prize_sub = pi_alpha[j];      // enter j
            } else {
                // BACKWARD: 新 label 在 j (新 head), 旧 head 是 L.node.
                // 原 forward arc is (j, L.node). Cost in forward sense = c[j, L.node].
                // enter L.node (旧 head) 现在不再是 head, 它的 prize 要计入新 label cost.
                arc_c = cost[(size_t)j * N + L.node];
                prize_sub = pi_alpha[L.node]; // 旧 head 的 prize (现在不再是 head)
            }
            double new_cost = L.cost + arc_c - prize_sub;
            const double new_cost_lb = subtract_binary64_down(
                add_binary64_down(S.cost_lb[lid], arc_c), prize_sub);
            const double new_cost_ub = subtract_binary64_up(
                add_binary64_up(S.cost_ub[lid], arc_c), prize_sub);

            // completion-LB prune on new label
            double lb_j = LB_other[j];
            if (lb_j < INF
                && add_binary64_down(new_cost_lb, lb_j)
                    > prune_threshold + EPS_LB_PRUNE) {
                ++S.n_pruned_lb;
                continue;
            }

            VisitedMask new_vis = L.visited;
            if (j < n_active) set_visited(new_vis, j);

            auto& bucket = S.per_node[j];

            // dominance: 是否被现存 dominate?
            bool dominated = false;
            for (size_t k = 0; k < bucket.size(); ++k) {
                if ((k & 255U) == 0U && timer && timer->expired()) {
                    S.timed_out = true;
                    break;
                }
                uint32_t eid = bucket[k];
                Label& E = S.labels[eid];
                if (!E.alive) continue;
                if (S.cost_ub[eid] <= new_cost_lb
                    && S.cap_ub[eid] <= new_cap_lb
                    && subset_of(E.visited, new_vis, n_words)) {
                    dominated = true;
                    break;
                }
            }
            if (S.timed_out) break;
            if (dominated) { ++S.n_pruned_dom; continue; }

            // create
            Label NL;
            NL.node = (uint32_t)j;
            NL.parent = lid;
            NL.visited = new_vis;
            NL.cap_used = new_cap;
            NL.cost = new_cost;
            NL.alive = true;
            uint32_t new_id = (uint32_t)S.labels.size();
            S.labels.push_back(NL);
            S.cost_lb.push_back(new_cost_lb);
            S.cost_ub.push_back(new_cost_ub);
            S.cap_lb.push_back(new_cap_lb);
            S.cap_ub.push_back(new_cap_ub);

            // 反向 kill: 用新 label dominate 现存
            const Label& Nref = S.labels[new_id];
            size_t write2 = 0;
            for (size_t k = 0; k < bucket.size(); ++k) {
                if ((k & 255U) == 0U && timer && timer->expired()) {
                    S.timed_out = true;
                    break;
                }
                uint32_t eid = bucket[k];
                Label& E = S.labels[eid];
                if (!E.alive) continue;
                bool weak = new_cost_ub <= S.cost_lb[eid]
                    && new_cap_ub <= S.cap_lb[eid]
                    && subset_of(Nref.visited, E.visited, n_words);
                if (weak && ( new_cost_ub < S.cost_lb[eid]
                           || new_cap_ub < S.cap_lb[eid]
                           || !equals_mask(Nref.visited, E.visited, n_words) )) {
                    E.alive = false;
                    ++S.n_pruned_dom;
                    continue;
                }
                bucket[write2++] = eid;
            }
            if (S.timed_out) break;
            bucket.resize(write2);
            bucket.push_back(new_id);
            work.push_back(new_id);

            if (label_budget && S.labels.size() >= label_budget) {
                S.budget_triggered = true;
                break;
            }
        }
        if (S.budget_triggered || S.timed_out) break;
    }
    return S;
}

inline std::vector<int> forward_path_from(const LabelStore& F, uint32_t fid) {
    std::vector<int> p;
    uint32_t cur = fid;
    while (cur != UINT32_MAX) {
        p.push_back((int)F.labels[cur].node);
        cur = F.labels[cur].parent;
    }
    std::reverse(p.begin(), p.end());
    return p;
}

// 还原 backward path: 从 B-label 沿 parent (parent = B 中"上一个 head", 即
// 原 forward 序列中"v 的 successor"). B label at v 的 parent 是 v 的下一个
// (在 v->...->de 顺序中). 所以从 v 开始, 一路 parent, 得 v -> ... -> de.
inline std::vector<int> backward_path_from(const LabelStore& B, uint32_t bid) {
    std::vector<int> p;
    uint32_t cur = bid;
    while (cur != UINT32_MAX) {
        p.push_back((int)B.labels[cur].node);
        cur = B.labels[cur].parent;
    }
    return p;  // 已经是 v -> ... -> de
}

// Join: 枚举 F@u + arc(u,v) + B@v. 返回 (cost, full_path) 候选, 已按 cost 升序.
//
// 处理边界:
//   - F at depot_end (单向 forward 走完, v 跳过 join arc): 通过 F@de + B@de
//     (root) 实现, arc 不存在 -> 特判.
//   - 实际我们把"forward 单独走完 path" 当 F@de 处理, 并与 B@de root pair
//     得到 cost = C_F, path = ds -> ... -> de (F 的 path).
//   - 同理 B-only path: B@ds + F@ds root.
//   - 中间 join: F@u (active u) × B@v (active v), u != v, arc (u,v).
struct JoinCandidate {
    double cost;
    double cap_total;  // q_F + q_B (for label-specific knapsack bound)
    uint32_t fid;
    uint32_t bid;
    int u;   // F head
    int v;   // B head
};

inline std::vector<JoinCandidate> join_candidates(
    const LabelStore& F, const LabelStore& B,
    const double* cost, const double* pi_alpha, const double* /*vol*/,
    int N, int /*n_active*/, int depot_start, int depot_end,
    double capacity, double prune_threshold,
    int n_words,
    const KnapsackPrep* knap_prep = nullptr,
    double pi_y = 0.0,
    double* incumbent_full = nullptr,
    const WallTimer* timer = nullptr,
    bool* timed_out = nullptr,
    bool sort_results = true)
{
    std::vector<JoinCandidate> cands;
    cands.reserve(1024);
    auto deadline_hit = [&]() -> bool {
        if (timer && timer->expired()) {
            if (timed_out) *timed_out = true;
            return true;
        }
        return false;
    };

    // 注意: prune_threshold 是 LB 剪枝用的 upper bound (常常 == greedy/incumbent),
    // 最优 path cost 可能恰好 == prune_threshold. 因此 join filter 要用 > + EPS,
    // 而不是 >=, 否则会把最优解过滤掉. (单向 si 版的 final filter 用 user_cutoff
    // 而不是 prune_threshold, 因此没这个问题.)
    const double JOIN_EPS = 1e-9;

    // (A) F-only paths (forward labels that reached depot_end):
    //     pair with B-root (B.labels[0] at depot_end with cost 0, V_B={})
    {
        const auto& fde = F.per_node[depot_end];
        for (size_t fpos = 0; fpos < fde.size(); ++fpos) {
            if ((fpos & 255U) == 0U && deadline_hit()) return cands;
            uint32_t fid = fde[fpos];
            if (!F.labels[fid].alive) continue;
            double cf = F.labels[fid].cost;
            const double cf_lb = F.cost_lb[fid];
            if (cf_lb > prune_threshold + JOIN_EPS) continue;
            double cap_tot = F.labels[fid].cap_used;
            const double cap_tot_lb = F.cap_lb[fid];
            if (cap_tot_lb > capacity) continue;
            if (knap_prep && incumbent_full) {
                double inactive_ub = knapsack_ub(
                    *knap_prep,
                    remaining_capacity_up(capacity, cap_tot_lb));
                double lb_full = pctsp_full_lower_bound(
                    cf_lb, inactive_ub, pi_y);
                if (lb_full > *incumbent_full + JOIN_EPS) continue;
            }
            JoinCandidate jc;
            jc.cost = cf;
            jc.cap_total = cap_tot;
            jc.fid = fid;
            jc.bid = 0;          // B root at depot_end
            jc.u = depot_end;
            jc.v = -1;           // sentinel: no join arc, F goes all the way
            cands.push_back(jc);
        }
    }

    // (B) B-only paths (backward labels that reached depot_start):
    //     pair with F-root (F.labels[0] at depot_start with cost 0, V_F={})
    {
        const auto& bds = B.per_node[depot_start];
        for (size_t bpos = 0; bpos < bds.size(); ++bpos) {
            if ((bpos & 255U) == 0U && deadline_hit()) return cands;
            uint32_t bid = bds[bpos];
            if (!B.labels[bid].alive) continue;
            // B-label at depot_start means full path ds -> ... -> de stored
            // in B form. cost = sum (c[a,b] - pi_alpha[b]) for arcs in path,
            // 不含 pi_alpha[ds]=0 anyway. = full forward path cost.
            double cb = B.labels[bid].cost;
            const double cb_lb = B.cost_lb[bid];
            if (cb_lb > prune_threshold + JOIN_EPS) continue;
            double cap_tot = B.labels[bid].cap_used;
            const double cap_tot_lb = B.cap_lb[bid];
            if (cap_tot_lb > capacity) continue;
            if (knap_prep && incumbent_full) {
                double inactive_ub = knapsack_ub(
                    *knap_prep,
                    remaining_capacity_up(capacity, cap_tot_lb));
                double lb_full = pctsp_full_lower_bound(
                    cb_lb, inactive_ub, pi_y);
                if (lb_full > *incumbent_full + JOIN_EPS) continue;
            }
            JoinCandidate jc;
            jc.cost = cb;
            jc.cap_total = cap_tot;
            jc.fid = 0;          // F root at depot_start
            jc.bid = bid;
            jc.u = -1;
            jc.v = depot_start;  // sentinel: no join arc, B goes all the way
            cands.push_back(jc);
        }
    }

    // (C) Middle join: F@u + arc(u,v) + B@v.
    //     Step 7: Pareto frontier on backward labels for faster pruning.
    //     For each node v, build sorted backward labels by cap_used ascending,
    //     with a running min of cost (Pareto frontier: min cost for cap <= r).
    struct BPareto {
        double cap_used;
        double cost;
        double prefix_min_cost;
        uint32_t bid;
    };
    // Pre-build Pareto frontiers for backward labels per node
    std::vector<std::vector<BPareto>> b_pareto(N);
    for (int v = 0; v < N; ++v) {
        if (deadline_hit()) return cands;
        if (v == depot_start) continue;
        const auto& bb = B.per_node[v];
        if (bb.empty()) continue;
        std::vector<BPareto> sorted;
        sorted.reserve(bb.size());
        for (size_t bpos = 0; bpos < bb.size(); ++bpos) {
            if ((bpos & 255U) == 0U && deadline_hit()) return cands;
            uint32_t bid = bb[bpos];
            if (!B.labels[bid].alive) continue;
            sorted.push_back({B.cap_lb[bid], B.labels[bid].cost,
                              B.cost_lb[bid], bid});
        }
        // Sort by cap_used ascending
        std::sort(sorted.begin(), sorted.end(),
                  [](const BPareto& a, const BPareto& b) {
                      return a.cap_used < b.cap_used;
                  });
        // Build Pareto: running min of cost
        if (!sorted.empty()) {
            double run_min = sorted[0].prefix_min_cost;
            sorted[0].prefix_min_cost = run_min;
            for (size_t k = 1; k < sorted.size(); ++k) {
                if (sorted[k].prefix_min_cost < run_min) {
                    run_min = sorted[k].prefix_min_cost;
                }
                sorted[k].prefix_min_cost = run_min;
            }
        }
        b_pareto[v] = std::move(sorted);
    }

    for (int u = 0; u < N; ++u) {
        if (deadline_hit()) return cands;
        if (u == depot_end) continue;
        const auto& fb = F.per_node[u];
        if (fb.empty()) continue;
        for (int v = 0; v < N; ++v) {
            if (deadline_hit()) return cands;
            if (v == u) continue;
            if (v == depot_start) continue;
            if (u == depot_start && v == depot_end) continue;
            const auto& bpv = b_pareto[v];
            if (bpv.empty()) continue;
            double arc_c = cost[(size_t)u * N + v];
            double prize_v = pi_alpha[v];
            double arc_red = arc_c - prize_v;
            const double arc_red_lb = subtract_binary64_down(
                arc_c, prize_v);

            for (size_t fpos = 0; fpos < fb.size(); ++fpos) {
                if ((fpos & 63U) == 0U && deadline_hit()) return cands;
                uint32_t fid = fb[fpos];
                const Label& Lf = F.labels[fid];
                if (!Lf.alive) continue;
                const double rem_cap_for_b = remaining_capacity_up(
                    capacity, F.cap_lb[fid]);
                // Exact capacity-aware prefix query.  The frontier ignores
                // visited-set overlap, so it is a relaxation and therefore a
                // safe lower bound for every join that remains feasible.
                auto feasible_end = std::upper_bound(
                    bpv.begin(), bpv.end(), rem_cap_for_b,
                    [](double cap, const BPareto& bp) {
                        return cap < bp.cap_used;
                    });
                if (feasible_end == bpv.begin()) continue;
                const double min_b_cost =
                    std::prev(feasible_end)->prefix_min_cost;
                double best_possible = add_binary64_down(
                    add_binary64_down(F.cost_lb[fid], arc_red_lb),
                    min_b_cost);
                if (best_possible > prune_threshold + JOIN_EPS) continue;

                for (size_t bpos = 0; bpos < bpv.size(); ++bpos) {
                    if ((bpos & 255U) == 0U && deadline_hit()) return cands;
                    const BPareto& bp = bpv[bpos];
                    // bpv is sorted by cap_used ascending.  Once this label
                    // exceeds the remaining capacity, every later label does
                    // too; breaking is exactly equivalent to the old scan.
                    if (bp.cap_used > rem_cap_for_b) break;
                    const Label& Lb = B.labels[bp.bid];
                    double q_tot = Lf.cap_used + Lb.cap_used;
                    const double q_tot_lb = add_binary64_down(
                        F.cap_lb[fid], B.cap_lb[bp.bid]);
                    if (q_tot_lb > capacity) continue;
                    // elementary: V_F ∩ V_B = {}
                    bool ovl = false;
                    for (int w = 0; w < n_words; ++w) {
                        if ((Lf.visited[w] & Lb.visited[w]) != 0ULL) {
                            ovl = true; break;
                        }
                    }
                    if (ovl) continue;
                    double tot = Lf.cost + arc_red + Lb.cost;
                    const double tot_lb = add_binary64_down(
                        add_binary64_down(F.cost_lb[fid], arc_red_lb),
                        B.cost_lb[bp.bid]);
                    if (tot_lb > prune_threshold + JOIN_EPS) continue;
                    if (knap_prep && incumbent_full) {
                        double inactive_ub = knapsack_ub(
                            *knap_prep,
                            remaining_capacity_up(capacity, q_tot_lb));
                        double lb_full = pctsp_full_lower_bound(
                            tot_lb, inactive_ub, pi_y);
                        if (lb_full > *incumbent_full + JOIN_EPS) continue;
                    }
                    JoinCandidate jc;
                    jc.cost = tot;
                    jc.cap_total = q_tot;
                    jc.fid = fid;
                    jc.bid = bp.bid;
                    jc.u = u;
                    jc.v = v;
                    cands.push_back(jc);
                }
            }
        }
    }

    if (sort_results) {
        std::sort(cands.begin(), cands.end(),
                  [](const JoinCandidate& a, const JoinCandidate& b){
                      return a.cost < b.cost;
                  });
    }
    return cands;
}

// Reconstruct full path from JoinCandidate.
inline std::vector<int> reconstruct_full_path(
    const JoinCandidate& jc,
    const LabelStore& F, const LabelStore& B)
{
    std::vector<int> path;
    if (jc.v < 0) {
        // F-only
        return forward_path_from(F, jc.fid);
    }
    if (jc.u < 0) {
        // B-only
        return backward_path_from(B, jc.bid);
    }
    // middle join: ds -> ... -> u -> v -> ... -> de
    std::vector<int> fp = forward_path_from(F, jc.fid);
    std::vector<int> bp = backward_path_from(B, jc.bid);
    path.reserve(fp.size() + bp.size());
    for (int n : fp) path.push_back(n);
    for (int n : bp) path.push_back(n);
    return path;
}

// Sanity: compute max active vol (for q_split feasibility check).
inline double max_active_vol(const double* vol, int n_active) {
    double m = 0.0;
    for (int i = 0; i < n_active; ++i) {
        if (vol[i] > m) m = vol[i];
    }
    return m;
}

// ============================================================================
// solve_espprc_bi -- bidirectional version of solve_espprc_core.
// Returns same Result struct. Fall back to single-direction if max vol > cap/2.
// ============================================================================
Result solve_espprc_bi(
    const double* cost,
    const double* pi_alpha,
    const double* vol,
    int N, int n_active, int depot_start, int depot_end,
    double capacity, double cutoff, int top_k,
    uint64_t label_budget)
{
    Result res;
    const int n_words = (n_active + 63) / 64;
    if (top_k <= 0) top_k = 1;

    // q_bound = cap/2 + max_active_vol 保证对任何 feasible path (q_total <= cap)
    // 都存在 (u,v) split 使 q_F(u) <= q_split AND q_B(v) <= q_bound.
    // 证明: 取最大 m s.t. q_m <= q_split=cap/2 -> q_F(c_m)=q_m<=q_split<=q_bound;
    //   q_B(c_{m+1}) = q_K - q_m < q_K - cap/2 + vol[c_{m+1}] <= cap/2 + max_vol = q_bound.
    // 若 max_vol > cap (i.e. single-customer path 都 infeasible), 直接 fallback.
    const double q_split = multiply_nonnegative_up(capacity, 0.5);
    const double mv = max_active_vol(vol, n_active);
    if (mv > capacity) {
        res.status = -1;
        res.n_labels = 0;
        return res;
    }
    const double q_bound = add_binary64_up(q_split, mv);

    // Completion LBs
    std::vector<double> LB_to_de = compute_completion_lb(
        DIR_FORWARD, cost, pi_alpha, N, n_active, depot_start, depot_end);
    std::vector<double> LB_from_ds = compute_completion_lb(
        DIR_BACKWARD, cost, pi_alpha, N, n_active, depot_start, depot_end);

    // Greedy nearest-neighbor UB (cheap) as initial prune_threshold.
    double prune_threshold = std::isfinite(cutoff) ? cutoff : INF;
    {
        VisitedMask g_visited; clear_visited(g_visited);
        double g_cap = 0.0, g_cost = 0.0;
        int g_cur = depot_start;
        while (true) {
            int best_j = -1; double best_arc = INF;
            for (int j = 0; j < n_active; ++j) {
                if (is_visited(g_visited, j)) continue;
                if (g_cap + vol[j] > capacity) continue;
                double a = cost[(size_t)g_cur * N + j] - pi_alpha[j];
                if (a < best_arc) { best_arc = a; best_j = j; }
            }
            if (best_j < 0) break;
            if (best_arc >= 0.0 && g_cur != depot_start) break;
            set_visited(g_visited, best_j);
            g_cap += vol[best_j]; g_cost += best_arc; g_cur = best_j;
        }
        g_cost += cost[(size_t)g_cur * N + depot_end];
        if (g_cost < prune_threshold) prune_threshold = g_cost;
    }

    // Run both directions
    LabelStore F = labeling_pass(DIR_FORWARD, cost, pi_alpha, vol,
                                 N, n_active, depot_start, depot_end,
                                 q_bound, LB_to_de, prune_threshold,
                                 label_budget);
    LabelStore B = labeling_pass(DIR_BACKWARD, cost, pi_alpha, vol,
                                 N, n_active, depot_start, depot_end,
                                 q_bound, LB_from_ds, prune_threshold,
                                 label_budget);

    // Join
    std::vector<JoinCandidate> cands = join_candidates(
        F, B, cost, pi_alpha, vol,
        N, n_active, depot_start, depot_end,
        capacity, prune_threshold, n_words);

    res.n_labels = (uint64_t)(F.labels.size() + B.labels.size());
    res.n_pruned_dom = F.n_pruned_dom + B.n_pruned_dom;
    res.n_pruned_lb  = F.n_pruned_lb  + B.n_pruned_lb;

    if (cands.empty()) {
        res.status = (F.budget_triggered || B.budget_triggered) ? 2 : 1;
        return res;
    }

    // Build top-K paths. join_candidates 已按 cost 升序排好且已剔除 > prune_threshold+EPS,
    // 这里再用 user-supplied cutoff 做最终 filter (与 si 版语义一致).
    res.paths.reserve(std::min((int)cands.size(), top_k));
    int kept = 0;
    for (const JoinCandidate& jc : cands) {
        if (kept >= top_k) break;
        if (std::isfinite(cutoff) && jc.cost >= cutoff) break;
        std::vector<int> p = reconstruct_full_path(jc, F, B);
        if (p.size() < 2) continue;
        res.paths.emplace_back(jc.cost, std::move(p));
        ++kept;
    }
    res.status = (F.budget_triggered || B.budget_triggered) ? 2 : 0;
    return res;
}

// ============================================================================

Result solve_espprc_core(
    const double* cost,            // (N, N) row-major
    const double* pi_alpha,        // (N,)
    const double* vol,             // (N,)
    int N,
    int n_active,
    int depot_start,
    int depot_end,
    double capacity,
    double cutoff,
    int top_k,
    uint64_t label_budget          // 硬上限, 防止状态爆炸 (0 = 不限)
) {
    if (n_active > MAX_ACTIVE_BITS) {
        throw std::runtime_error("espprc: n_active>" + std::to_string(MAX_ACTIVE_BITS)
                                 + " (visited bitset capacity exceeded), 当前="
                                 + std::to_string(n_active));
    }
    const int n_words = (n_active + 63) / 64;
    if (top_k <= 0) top_k = 1;

    std::vector<Label> labels;
    labels.reserve(1 << 16);

    // 每个 node 维护 alive label id list (dominance check 时遍历)
    std::vector<std::vector<uint32_t>> per_node(N);
    for (auto& v : per_node) v.reserve(64);

    // BFS-style work queue (label id)
    std::deque<uint32_t> work;

    // 终点 candidate (depot_end 上的 final-cost label id)
    std::vector<uint32_t> finals;

    // 用户传入的 path-cost 上限 (作为 hard cutoff, FINAL filter 时用; NOT
    // 用于中间 label 的 monotone pruning, 因为 reduced cost = c - π_α 可负,
    // 一个 label 的 cost 完全可能在后续 extension 中变更小).
    const double user_cutoff = cutoff;

    // Root: depot_start, 无 prize (pi_alpha[depot_start]=0 by convention)
    {
        Label root;
        root.node = (uint32_t)depot_start;
        root.parent = UINT32_MAX;
        clear_visited(root.visited);
        root.cap_used = 0.0;
        root.cost = 0.0;
        root.alive = true;
        labels.push_back(root);
        per_node[depot_start].push_back(0);
        work.push_back(0);
    }

    // -----------------------------------------------------------------------
    // 预计算 completion lower bound: LB_final[j] = relaxed min cost from j to
    // depot_end. Relaxed = 允许非 elementary (访问重复节点) + 忽略 capacity.
    //
    // 这是真 elementary 路径成本的 SAFE 下界 (relaxation 必然 <=). 用来配合
    // top-K best 做安全剪枝: 若 L.cost + LB_final[L.node] >= kth_best, 这个
    // label 无论怎么扩展都不可能进 top-K, 直接剪.
    //
    // 计算方式: LB_h[h][j] = "j 出发用 ≤ h 弧到 depot_end 的最小 relaxed
    // path cost". 用 Bellman-style 递推. max_hops 取 n_active+1 (elementary
    // path 最长就这么多弧, 多了一定违反 elementary).
    //
    // 注意: arc cost = c[i,j] - π_α[j] 可负, 所以 LB_h[h] 随 h 单增不一定
    // (longer path 可能更便宜), 最后我们取 min_h LB_h[h][j].
    // -----------------------------------------------------------------------
    std::vector<double> LB_final(N, INF);
    {
        const int max_hops = n_active + 1;  // elementary path 弧数 ≤ n_active+1
        std::vector<double> prev(N, INF);
        std::vector<double> curr(N, INF);
        prev[depot_end] = 0.0;                 // h=0: 唯一可达 depot_end 自身
        for (int j = 0; j < N; ++j) {
            if (prev[j] < LB_final[j]) LB_final[j] = prev[j];
        }
        for (int h = 1; h <= max_hops; ++h) {
            for (int j = 0; j < N; ++j) {
                double best = INF;
                if (j == depot_end) { curr[j] = 0.0; continue; }
                // 尝试 j -> m, 再用 prev[m] (h-1 步到 depot_end)
                for (int m = 0; m < N; ++m) {
                    if (m == j) continue;
                    if (m == depot_start) continue;  // 不能回 depot_start
                    if (prev[m] >= INF) continue;
                    const double arc = subtract_binary64_down(
                        cost[(size_t)j * N + m], pi_alpha[m]);
                    const double tot = add_binary64_down(arc, prev[m]);
                    if (tot < best) best = tot;
                }
                curr[j] = best;
            }
            for (int j = 0; j < N; ++j) {
                if (curr[j] < LB_final[j]) LB_final[j] = curr[j];
            }
            std::swap(prev, curr);
        }
    }

    // -----------------------------------------------------------------------
    // Single-best 追踪 (用于剪枝): prune_threshold = 已知最佳 final cost (含
    // greedy UB seed + user_cutoff). Note: 这意味着 top_k>1 时 LB 剪枝可能丢
    // 次优解 (≥ 当前最佳的). 上层 caller 要求严格 OPT (LSBC), 第 2..top_k
    // 仅用于 plan_pool 加速, 丢了不影响 correctness.
    //
    // Greedy UB seed: 在 main loop 前用 nearest-neighbor (按 reduced cost 升)
    // 搞一条 feasible elementary path, 作为 prune_threshold 的初值. 这样
    // LB 剪枝从一开始就 fire, 不等到第一个 final 到达 depot_end.
    // -----------------------------------------------------------------------
    double prune_threshold = std::isfinite(user_cutoff) ? user_cutoff : INF;

    // -- Greedy nearest-neighbor (reduced cost) UB --
    // 从 depot_start 出发, 每步贪心选 unvisited active j 中 (c[cur,j]-π_α[j])
    // 最小者 (允许负, 当然越负越好). 不要求改善总 cost; 只要 feasible.
    // 走完或没可行扩展时关到 depot_end.
    {
        VisitedMask g_visited;
        clear_visited(g_visited);
        double g_cap = 0.0;
        double g_cost = 0.0;
        int g_cur = depot_start;
        while (true) {
            int best_j = -1;
            double best_arc = INF;
            for (int j = 0; j < n_active; ++j) {
                if (is_visited(g_visited, j)) continue;
                if (g_cap + vol[j] > capacity) continue;
                double a = cost[(size_t)g_cur * N + j] - pi_alpha[j];
                if (a < best_arc) {
                    best_arc = a;
                    best_j = j;
                }
            }
            // 只要还有 negative arc 就走 (能降 cost 就走);
            // positive arc 时也可以走以争取更紧 UB, 但容易走太远不优, 这里
            // 启发式: 只接受 negative 或者首次扩展.
            if (best_j < 0) break;
            if (best_arc >= 0.0 && g_cur != depot_start) break;
            set_visited(g_visited, best_j);
            g_cap += vol[best_j];
            g_cost += best_arc;
            g_cur = best_j;
        }
        // 关到 depot_end
        g_cost += cost[(size_t)g_cur * N + depot_end];
        if (g_cost < prune_threshold) prune_threshold = g_cost;
    }

    auto register_final = [&](uint32_t lid) {
        finals.push_back(lid);
        double c = labels[lid].cost;
        if (c < prune_threshold) prune_threshold = c;
    };
    // 把 kth_best 用作 alias, 后面代码不用改
    double& kth_best = prune_threshold;

    uint64_t n_pruned_dom = 0;
    uint64_t n_pruned_lb = 0;   // 诊断: 被 completion LB 剪掉的 label 数
    bool cutoff_triggered = false;

    while (!work.empty()) {
        uint32_t lid = work.front();
        work.pop_front();
        if (!labels[lid].alive) continue;
        // 拷贝, 后面 push_back 会重新 alloc labels[]
        Label L = labels[lid];

        // depot_end: 注册 final, 更新 top-K, 不再扩展
        if ((int)L.node == depot_end) {
            register_final(lid);
            continue;
        }

        // 安全 completion-LB 剪枝: 若 L.cost + LB(L.node) >= kth_best, 这个
        // label 怎么扩展都不可能进 top-K (LB 是 relaxed 下界, 真实 elementary
        // 完成成本 >= LB).
        if (L.cost + LB_final[L.node] > kth_best + EPS_LB_PRUNE) {
            ++n_pruned_lb;
            continue;
        }

        // 扩展到所有可能的下一节点
        for (int j = 0; j < N; ++j) {
            if (j == (int)L.node) continue;
            if (j == depot_start) continue;  // 不能回 depot_start
            // elementary: active customer 只能访问一次
            if (j < n_active) {
                if (is_visited(L.visited, j)) continue;
            }
            // capacity (只 active 客户算)
            double new_cap = L.cap_used;
            if (j < n_active) new_cap += vol[j];
            if (new_cap > capacity) continue;

            double arc_c = cost[(size_t)L.node * N + j];
            double prize = pi_alpha[j];  // depot 都为 0
            double new_cost = L.cost + arc_c - prize;

            // 安全 completion-LB 剪枝 (作用于将要插入的新 label).
            if (new_cost + LB_final[j] > kth_best + EPS_LB_PRUNE) {
                ++n_pruned_lb;
                continue;
            }

            VisitedMask new_vis = L.visited;
            if (j < n_active) set_visited(new_vis, j);

            auto& bucket = per_node[j];

            // 1) 看新 label 是否被现存 dominate. early break, 不 compact 死 label
            //    (这里若把 compaction 也做了, 就丢了 break, 反而显著拖慢热路径).
            bool dominated = false;
            for (size_t k = 0; k < bucket.size(); ++k) {
                uint32_t eid = bucket[k];
                Label& E = labels[eid];
                if (!E.alive) continue;
                if (E.cost <= new_cost
                    && E.cap_used <= new_cap
                    && subset_of(E.visited, new_vis, n_words)) {
                    dominated = true;
                    break;
                }
            }
            if (dominated) { ++n_pruned_dom; continue; }

            // 2) 创建新 label.
            Label NL;
            NL.node = (uint32_t)j;
            NL.parent = lid;
            NL.visited = new_vis;
            NL.cap_used = new_cap;
            NL.cost = new_cost;
            NL.alive = true;
            uint32_t new_id = (uint32_t)labels.size();
            labels.push_back(NL);
            // 注意: labels.push_back 可能 realloc, 上面拿 L 已经拷贝过, 安全

            // 3) 用新 label 反过来 kill 被它严格 dominate 的现存. 这趟无论如何
            //    都要扫全桶, 顺便 compact 出死 label 不增加复杂度.
            const Label& Nref = labels[new_id];
            size_t write2 = 0;
            for (size_t k = 0; k < bucket.size(); ++k) {
                uint32_t eid = bucket[k];
                Label& E = labels[eid];
                if (!E.alive) continue;  // 死 label: 顺手 compact 掉
                bool weak = Nref.cost <= E.cost
                    && Nref.cap_used <= E.cap_used
                    && subset_of(Nref.visited, E.visited, n_words);
                if (weak && ( Nref.cost < E.cost
                           || Nref.cap_used < E.cap_used
                           || !equals_mask(Nref.visited, E.visited, n_words) )) {
                    E.alive = false;
                    ++n_pruned_dom;
                    continue;  // 杀掉后也不写回
                }
                bucket[write2++] = eid;
            }
            bucket.resize(write2);

            bucket.push_back(new_id);
            work.push_back(new_id);

            if (label_budget && labels.size() >= label_budget) {
                cutoff_triggered = true;
                break;  // 内层 break, 外层 while 还会跑完 queue, 但不创建新 label
            }
        }
        if (cutoff_triggered) break;
    }

    Result res;
    res.n_labels = (uint64_t)labels.size();
    res.n_pruned_dom = n_pruned_dom;
    res.n_pruned_lb = n_pruned_lb;

    if (finals.empty()) {
        res.status = cutoff_triggered ? 2 : 1;
        return res;
    }

    // 排序按 cost
    std::sort(finals.begin(), finals.end(),
              [&](uint32_t a, uint32_t b) { return labels[a].cost < labels[b].cost; });

    // 按 user_cutoff 在最后过滤 (path cost >= user_cutoff 直接丢)
    res.paths.reserve(std::min((int)finals.size(), top_k));
    int kept = 0;
    for (size_t i = 0; i < finals.size() && kept < top_k; ++i) {
        uint32_t lid = finals[i];
        double final_cost = labels[lid].cost;
        if (final_cost >= user_cutoff) break;  // 已排序, 后面更差
        std::vector<int> path;
        uint32_t cur = lid;
        while (cur != UINT32_MAX) {
            path.push_back((int)labels[cur].node);
            cur = labels[cur].parent;
        }
        std::reverse(path.begin(), path.end());
        res.paths.emplace_back(final_cost, std::move(path));
        ++kept;
    }
    res.status = cutoff_triggered ? 2 : 0;
    return res;
}

inline void validate_finite_array(const py::buffer_info& buf,
                                  const char* name) {
    const double* values = static_cast<const double*>(buf.ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i) {
        if (!std::isfinite(values[i])) {
            throw std::runtime_error(
                std::string(name) + " must contain only finite values");
        }
    }
}

inline void validate_nonnegative_array(const py::buffer_info& buf,
                                       const char* name) {
    const double* values = static_cast<const double*>(buf.ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i) {
        if (!std::isfinite(values[i]) || values[i] < 0.0) {
            throw std::runtime_error(
                std::string(name)
                + " must contain only finite nonnegative values");
        }
    }
}

inline void validate_cutoff(double cutoff) {
    if (std::isnan(cutoff) || cutoff == -INF) {
        throw std::runtime_error("cutoff must be finite or +inf");
    }
}

py::dict solve_espprc_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> cost_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> pi_alpha_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> vol_arr,
    int n_active,
    int depot_start,
    int depot_end,
    double capacity,
    double cutoff,
    int top_k,
    uint64_t label_budget,
    bool use_bidirectional
) {
    auto c_buf = cost_arr.request();
    auto pi_buf = pi_alpha_arr.request();
    auto vol_buf = vol_arr.request();
    if (c_buf.ndim != 2 || c_buf.shape[0] != c_buf.shape[1]) {
        throw std::runtime_error("cost 必须是 (N,N) 方阵");
    }
    int N = (int)c_buf.shape[0];
    if (pi_buf.ndim != 1 || vol_buf.ndim != 1
        || (int)pi_buf.shape[0] != N || (int)vol_buf.shape[0] != N) {
        throw std::runtime_error("pi_alpha / vol 长度必须等于 N");
    }
    if (depot_start < 0 || depot_start >= N || depot_end < 0 || depot_end >= N
        || depot_start == depot_end) {
        throw std::runtime_error("depot indices 不合法");
    }
    if (n_active < 0 || n_active > N - 2) {
        throw std::runtime_error("n_active 不合法");
    }
    if (N != n_active + 2
        || depot_start < n_active || depot_end < n_active) {
        throw std::runtime_error(
            "ESPPRC layout must be active customers followed by two depots");
    }
    if (!std::isfinite(capacity) || capacity < 0.0) {
        throw std::runtime_error("capacity must be finite and nonnegative");
    }
    validate_cutoff(cutoff);
    validate_finite_array(c_buf, "cost");
    validate_finite_array(pi_buf, "pi_alpha");
    validate_nonnegative_array(vol_buf, "vol");

    Result r;
    bool used_bi = false;
    {
        py::gil_scoped_release nogil;
        if (use_bidirectional) {
            r = solve_espprc_bi(
                (const double*)c_buf.ptr,
                (const double*)pi_buf.ptr,
                (const double*)vol_buf.ptr,
                N, n_active, depot_start, depot_end,
                capacity,
                std::isfinite(cutoff) ? cutoff : INF,
                top_k, label_budget);
            if (r.status == -1) {
                // fallback: max vol > cap/2, bidirectional infeasible
                r = solve_espprc_core(
                    (const double*)c_buf.ptr,
                    (const double*)pi_buf.ptr,
                    (const double*)vol_buf.ptr,
                    N, n_active, depot_start, depot_end,
                    capacity,
                    std::isfinite(cutoff) ? cutoff : INF,
                    top_k, label_budget);
            } else {
                used_bi = true;
            }
        } else {
            r = solve_espprc_core(
                (const double*)c_buf.ptr,
                (const double*)pi_buf.ptr,
                (const double*)vol_buf.ptr,
                N, n_active, depot_start, depot_end,
                capacity,
                std::isfinite(cutoff) ? cutoff : INF,
                top_k, label_budget);
        }
    }

    py::list paths;
    for (auto& pc : r.paths) {
        py::list seq;
        for (int n : pc.second) seq.append(n);
        paths.append(py::make_tuple(pc.first, seq));
    }
    py::dict out;
    out["paths"] = paths;
    out["status"] = r.status;
    out["n_labels"] = r.n_labels;
    out["n_pruned_dom"] = r.n_pruned_dom;
    out["n_pruned_lb"] = r.n_pruned_lb;
    out["used_bi"] = used_bi;
    return out;
}

// ============================================================================
// Certifying PCTSP core: ng-route labeling + decremental state-space
// relaxation (DSSR) with capacity-indexed completion bounds.
//
//   Feillet, Dejax, Gendreau, Gueguen (2004)   -- ESPPRC labeling / dominance
//   Righini & Salani (2008)                    -- DSSR
//   Baldacci, Mingozzi & Roberti (2011)        -- ng-route relaxation
//   Christofides, Mingozzi & Toth (1981)       -- q-routes, 2-cycle elimination
//
// Every DSSR iteration solves one ng-route relaxation exactly with a best-first
// labeling (A*-style termination on the certified key).  Its optimum is a
// certified lower bound on the elementary optimum.  When that optimum is
// attained by an elementary policy the policy is optimal; otherwise the
// customers repeated by the relaxed optimum become "critical" (never forgotten)
// and the next iteration is a strictly tighter relaxation.  Interruption keeps
// a certified bound: max(previous completed relaxation, current frontier).
//
// Exactness conventions are unchanged from the historical core: every reported
// bound is a directed binary64 endpoint, incumbent objectives are exact dyadic
// sums, and dominance on cost is decided on directed intervals with an exact
// fallback.  Loads are exact 128-bit scaled integers whenever the binary64
// volume domain permits (always true for production data).
// ============================================================================

    struct DirectedArcEndpoints {
        double lower;
        double upper;
    };
    static_assert(sizeof(DirectedArcEndpoints) == 2 * sizeof(double),
                  "directed arc endpoints must remain tightly packed");

inline uint64_t mask_and(uint64_t a, uint64_t b) { return a & b; }
inline uint64_t mask_or(uint64_t a, uint64_t b) { return a | b; }
inline int mask_popcount(uint64_t a) { return __builtin_popcountll(a); }

inline VisitedMask mask_and(const VisitedMask& a, const VisitedMask& b) {
    VisitedMask r;
    for (int w = 0; w < MAX_VISITED_WORDS; ++w) r[w] = a[w] & b[w];
    return r;
}
inline VisitedMask mask_or(const VisitedMask& a, const VisitedMask& b) {
    VisitedMask r;
    for (int w = 0; w < MAX_VISITED_WORDS; ++w) r[w] = a[w] | b[w];
    return r;
}
inline int mask_popcount(const VisitedMask& a) {
    int c = 0;
    for (int w = 0; w < MAX_VISITED_WORDS; ++w) c += __builtin_popcountll(a[w]);
    return c;
}

// Exact detour rule.  Customer j is dropped when every policy visiting j is
// strictly improved by skipping it.  Interior visit i->j->k: the shortcut saves
// cost(i,j)+cost(j,k)-cost(i,k) and forfeits pi(j).  Single-customer route
// (ds,j,de): the alternatives are y=0 (value 0) or, with inactive items, a
// knapsack-only y=1 policy, so the route must cost more than max(pi_y, 0).
// All arithmetic is rounded toward "keep", so the optimal set is unchanged and
// the tie-break among optimal policies is preserved (strict inequalities).
inline std::vector<char> detour_dominated_customers(
    const double* cost, const double* pi_alpha, const double* vol,
    int N, int n_active, int depot_start, int depot_end,
    double capacity, double pi_y)
{
    std::vector<char> drop(n_active, 0);
    const double single_threshold = std::max(pi_y, 0.0);
    for (int j = 0; j < n_active; ++j) {
        if (vol[j] > capacity) { drop[j] = 1; continue; }
        const double pj = pi_alpha[j];
        bool dominated = true;
        for (int i = 0; i < N && dominated; ++i) {
            if (i == j || i == depot_end) continue;
            const double c_ij = cost[(size_t)i * N + j];
            for (int k = 0; k < N; ++k) {
                if (k == j || k == i || k == depot_start) continue;
                const double through = add_binary64_down(
                    c_ij, cost[(size_t)j * N + k]);
                if (i == depot_start && k == depot_end) {
                    if (!(subtract_binary64_down(through, pj)
                          > single_threshold)) {
                        dominated = false;
                        break;
                    }
                    continue;
                }
                const double detour = subtract_binary64_down(
                    through, cost[(size_t)i * N + k]);
                if (!(detour > pj)) {
                    dominated = false;
                    break;
                }
            }
        }
        drop[j] = dominated ? 1 : 0;
    }
    return drop;
}

// ng-set of j: j itself plus its ng_size nearest kept customers (symmetric
// routing distance).  A label at j forgets every customer outside N_j.
// `order[j]` lists every other kept customer by increasing distance from j, so
// the completion bound's smaller neighbourhood is always a prefix of N_j.
template <typename Mask>
struct NgSets {
    std::vector<Mask> mask;
    std::vector<std::vector<int>> order;
};

template <typename Mask>
inline NgSets<Mask> build_ng_sets(
    const double* cost, int N, int n_active,
    const std::vector<char>& drop, int ng_size)
{
    NgSets<Mask> ng;
    ng.mask.resize(n_active);
    ng.order.resize(n_active);
    std::vector<std::pair<double, int>> order;
    order.reserve(n_active);
    for (int j = 0; j < n_active; ++j) {
        Mask m;
        clear_visited(m);
        set_visited(m, j);
        if (!drop[j]) {
            order.clear();
            for (int k = 0; k < n_active; ++k) {
                if (k == j || drop[k]) continue;
                order.emplace_back(
                    cost[(size_t)j * N + k] + cost[(size_t)k * N + j], k);
            }
            std::sort(order.begin(), order.end());
            ng.order[j].reserve(order.size());
            for (int t = 0; t < (int)order.size(); ++t) {
                ng.order[j].push_back(order[t].second);
                if (t < ng_size) set_visited(m, order[t].second);
            }
        }
        ng.mask[j] = m;
    }
    return ng;
}

// Dominance-archive slot keys: the ng-part of a label's memory restricted to
// the first SLOT_KEY_BITS members of N_j.  SLOT_SUBSETS[k] / SLOT_SUPERSETS[k]
// are bitsets over all 2^SLOT_KEY_BITS keys marking the subsets / supersets
// of k, so the slots a label must be compared against are found by ANDing
// with the node's non-empty-slot bitset.
constexpr int SLOT_KEY_BITS = 8;
constexpr int SLOT_COUNT = 1 << SLOT_KEY_BITS;
constexpr int SLOT_WORDS = SLOT_COUNT / 64;
struct SlotSet {
    uint64_t w[SLOT_WORDS] = {};
};
inline std::array<SlotSet, SLOT_COUNT> build_slot_relation(bool subsets) {
    std::array<SlotSet, SLOT_COUNT> table{};
    for (uint32_t k = 0; k < (uint32_t)SLOT_COUNT; ++k) {
        for (uint32_t s = 0; s < (uint32_t)SLOT_COUNT; ++s) {
            const bool rel = subsets ? ((s & ~k) == 0u) : ((k & ~s) == 0u);
            if (rel) table[k].w[s >> 6] |= 1ULL << (s & 63u);
        }
    }
    return table;
}
const std::array<SlotSet, SLOT_COUNT> SLOT_SUBSETS = build_slot_relation(true);
const std::array<SlotSet, SLOT_COUNT> SLOT_SUPERSETS = build_slot_relation(false);

// ng-route completion bound (Baldacci-Mingozzi-Roberti style DP).
//
// State (r, j, M): a walk sits at kept customer j remembering the set M of
// customers (M subset of B_j, the bound neighbourhood of j; j itself is
// implicit) and may still spend r capacity units.  f(r, j, M) is a
// directed-down lower bound on the reduced cost of any 2-cycle-free ng-walk
// j -> ... -> depot_end that (i) never enters a remembered customer, where
// memory evolves as M' = (M u {j}) n B_m on the arc j -> m, (ii) never
// immediately returns (best/second-best recurrence), and (iii) whose visits
// weigh at most r units in total.
//
// Validity for the labeling: B_j is a prefix of the labeling neighbourhood
// N_j and a label's memory only contains visited customers, so every
// elementary completion of a label at j with memory `mem` is feasible for the
// DP from state (r, j, mem n B_j) as soon as r bounds its total weight.  The
// bound is monotone: more memory or fewer units never lowers it.  Together
// with mem-subset / load dominance this is exactly what the certifying
// argument in solve_pctsp_ng_impl needs.
//
// The table is built at escalating precision levels (capacity resolution and
// neighbourhood size); the labeling starts cheap and rebuilds a stronger
// table only when the search turns out to need it.
constexpr int MAX_BOUND_NG = 8;

struct BoundLevel {
    int cap_bits;   // capacity / unit in [2^cap_bits, 2^(cap_bits+1))
    int ng_b;       // bound neighbourhood size (capped by the work budget)
};
#ifndef PCTSP_BOUND_LEVELS
#define PCTSP_BOUND_LEVELS {{6, 3}, {7, 6}, {8, MAX_BOUND_NG}}
#endif
constexpr BoundLevel BOUND_LEVELS[] = PCTSP_BOUND_LEVELS;
constexpr int N_BOUND_LEVELS = (int)(sizeof(BOUND_LEVELS) / sizeof(BOUND_LEVELS[0]));

inline uint64_t bound_transitions(uint64_t rows, uint64_t n_kept, int ng_b) {
    return rows * n_kept * (1ULL << ng_b) * n_kept;
}
inline int bound_ng_for_budget(uint64_t rows, int n_kept, int ng_b_requested,
                               uint64_t work_budget) {
    int ng_b = std::min({ng_b_requested, MAX_BOUND_NG, n_kept - 1});
    if (ng_b < 0) ng_b = 0;
    while (ng_b > 0 && bound_transitions(rows, n_kept, ng_b) > work_budget) {
        --ng_b;
    }
    return ng_b;
}

struct CompletionBound {
    int rows_max = 0;            // largest resource index
    int ng_b = 0;                // bound neighbourhood size actually used
    int S = 1;                   // 1 << ng_b memory states per customer
    int n_kept = 0;
    uint64_t transitions = 0;    // DP work actually performed (upper bound)
    std::vector<int> kidx;       // node -> kept index, -1 otherwise
    std::vector<std::array<int, MAX_BOUND_NG>> members;  // per kept index
    std::vector<int> n_members;  // per kept index
    // (rows_max+1) x n_kept x S, rounded towards -inf into binary32 so the
    // table stays small enough to be cache friendly during the labeling.
    std::vector<float> f;
    double root_lb = INF;

    template <typename Mask>
    int local_index(int j, const Mask& mem) const {
        const int k = kidx[j];
        int idx = 0;
        for (int t = 0; t < n_members[k]; ++t) {
            if (is_visited(mem, members[k][t])) idx |= (1 << t);
        }
        return idx;
    }
    const float* entry(int j, int idx, int r) const {
        return &f[((size_t)r * n_kept + kidx[j]) * S + idx];
    }
    double lookup(int j, int idx, int r) const {
        return (double)*entry(j, idx, r);
    }
};

inline float binary64_to_binary32_down(double x) {
    if (x >= INF) return std::numeric_limits<float>::infinity();
    float v = (float)x;
    if ((double)v > x) v = std::nextafterf(v, -std::numeric_limits<float>::infinity());
    return v;
}

// weight[node] >= 1 for every kept customer (so that rows strictly decrease
// along a walk and the DP is a single pass over increasing r).
// bound_members[j] lists the candidates for B_j in priority order; the first
// ng_b of them are used, where ng_b is the largest size within the budget.
inline CompletionBound build_completion_bound(
    const std::vector<int>& weight, int rows_max,
    const std::vector<DirectedArcEndpoints>& arc,
    int N, int n_active, int depot_start, int depot_end,
    const std::vector<char>& drop,
    const std::vector<std::vector<int>>& bound_members,
    int ng_b_requested, uint64_t work_budget)
{
    CompletionBound cb;
    cb.rows_max = std::max(rows_max, 0);
    cb.kidx.assign(N, -1);
    std::vector<int> kept;
    for (int j = 0; j < n_active; ++j) {
        if (drop[j]) continue;
        cb.kidx[j] = (int)kept.size();
        kept.push_back(j);
    }
    cb.n_kept = (int)kept.size();
    if (cb.n_kept == 0) return cb;
    const int n_kept = cb.n_kept;
    const size_t rows = (size_t)cb.rows_max + 1;

    const int ng_b = bound_ng_for_budget(rows, n_kept, ng_b_requested,
                                         work_budget);
    cb.ng_b = ng_b;
    cb.S = 1 << ng_b;
    const int S = cb.S;
    cb.members.assign(n_kept, {});
    cb.n_members.assign(n_kept, 0);
    std::vector<int8_t> pos((size_t)N * N, -1);   // pos[m*N+g]: slot of g in B_m
    for (int k = 0; k < n_kept; ++k) {
        const int j = kept[k];
        cb.members[k].fill(-1);
        const auto& ord = bound_members[j];
        for (int t = 0; t < ng_b && t < (int)ord.size(); ++t) {
            cb.members[k][t] = ord[t];
            pos[(size_t)j * N + ord[t]] = (int8_t)t;
            ++cb.n_members[k];
        }
    }
    // trans[(k*n_kept + km)*S + idx]: memory index at m after j -> m from
    // memory idx at j, or -1 when m is remembered (or m == j).
    std::vector<int16_t> trans((size_t)n_kept * n_kept * S, -1);
    for (int k = 0; k < n_kept; ++k) {
        const int j = kept[k];
        const int n_idx = 1 << cb.n_members[k];
        for (int km = 0; km < n_kept; ++km) {
            const int m = kept[km];
                    if (m == j) continue;
            int16_t* tr = &trans[((size_t)k * n_kept + km) * S];
            const int pj = pos[(size_t)m * N + j];
            const int base_m = (pj >= 0) ? (1 << pj) : 0;
            for (int idx = 0; idx < n_idx; ++idx) {
                bool remembered = false;
                int idx_m = base_m;
                for (int t = 0; t < cb.n_members[k]; ++t) {
                    if (!(idx & (1 << t))) continue;
                    const int g = cb.members[k][t];
                    if (g == m) { remembered = true; break; }
                    const int p = pos[(size_t)m * N + g];
                    if (p >= 0) idx_m |= (1 << p);
                }
                if (!remembered) tr[idx] = (int16_t)idx_m;
            }
        }
    }

    // Row-major (r, k, idx); for fixed (r, k, km) the inner idx loop reads one
    // contiguous S-segment of the predecessor row.
    std::vector<double> f1(rows * n_kept * S, INF);
    std::vector<double> f2(rows * n_kept * S, INF);
    std::vector<int16_t> s1(rows * n_kept * S, -1);
    std::vector<double> best1(S), best2(S);
    std::vector<int16_t> arg1(S);
    uint64_t transitions = 0;
    for (int r = 0; r <= cb.rows_max; ++r) {
        for (int k = 0; k < n_kept; ++k) {
            const int j = kept[k];
            const double to_depot = arc[(size_t)j * N + depot_end].lower;
            const int n_idx = 1 << cb.n_members[k];
            std::fill(best1.begin(), best1.begin() + n_idx, to_depot);
            std::fill(best2.begin(), best2.begin() + n_idx, INF);
            std::fill(arg1.begin(), arg1.begin() + n_idx, (int16_t)-1);
            for (int km = 0; km < n_kept; ++km) {
                const int m = kept[km];
                if (m == j) continue;
                const int wm = weight[m];
                if (wm > r) continue;
                const int16_t* tr = &trans[((size_t)k * n_kept + km) * S];
                const size_t prev_base = ((size_t)(r - wm) * n_kept + km) * S;
                const double* pf1 = &f1[prev_base];
                const double* pf2 = &f2[prev_base];
                const int16_t* ps1 = &s1[prev_base];
                const double a = arc[(size_t)j * N + m].lower;
                transitions += (uint64_t)n_idx;
                for (int idx = 0; idx < n_idx; ++idx) {
                    const int idx_m = tr[idx];
                    if (idx_m < 0) continue;
                    const double base = (ps1[idx_m] == j) ? pf2[idx_m]
                                                          : pf1[idx_m];
                    if (base >= INF) continue;
                    const double val = add_binary64_down(a, base);
                    if (val < best1[idx]) {
                        best2[idx] = best1[idx];
                        best1[idx] = val;
                        arg1[idx] = (int16_t)m;
                    } else if (val < best2[idx]) {
                        best2[idx] = val;
                    }
                }
            }
            const size_t cur = ((size_t)r * n_kept + k) * S;
            std::copy(best1.begin(), best1.begin() + n_idx, f1.begin() + cur);
            std::copy(best2.begin(), best2.begin() + n_idx, f2.begin() + cur);
            std::copy(arg1.begin(), arg1.begin() + n_idx, s1.begin() + cur);
        }
    }
    cb.transitions = transitions;
    // Root: depot_start remembers nothing, so the first customer m starts with
    // memory {m} (local index 0).
    for (int km = 0; km < n_kept; ++km) {
        const int m = kept[km];
        const int wm = weight[m];
        if (wm > cb.rows_max) continue;
        const size_t st = ((size_t)(cb.rows_max - wm) * n_kept + km) * S;
        if (f1[st] >= INF) continue;
        const double val = add_binary64_down(
            arc[(size_t)depot_start * N + m].lower, f1[st]);
        cb.root_lb = std::min(cb.root_lb, val);
    }
    cb.f.resize(f1.size());
    for (size_t i = 0; i < f1.size(); ++i) cb.f[i] = binary64_to_binary32_down(f1[i]);
    return cb;
}

// Discretised capacity for the q-indexed table.  The unit is a power of two,
// so floor(vol/unit) and floor(capacity/unit) are exact and rounding weights
// down only enlarges the relaxed set.  Customers whose weight rounds to zero
// get weight one and must be kept elementary ("critical") by the caller, so
// units_total = units + n_zero_weight bounds the total weight of every walk
// the labeling can generate.
struct CapacityUnits {
    double unit = 1.0;
    int units = 0;               // floor(capacity / unit)
    int n_zero_weight = 0;
    int units_total = 0;
    std::vector<int> weight;     // per node, >= 1 for kept customers
    std::vector<char> zero_weight;
};

inline CapacityUnits build_capacity_units(
    const double* vol, int N, int n_active, double capacity,
    const std::vector<char>& drop, int cap_bits)
{
    CapacityUnits cu;
    if (capacity > 0.0) {
        // capacity/unit in [2^cap_bits, 2^(cap_bits+1)).
        cu.unit = std::ldexp(1.0, std::ilogb(capacity) - cap_bits);
        cu.units = (int)std::floor(capacity / cu.unit);
    }
    cu.weight.assign(N, 0);
    cu.zero_weight.assign(N, 0);
    for (int j = 0; j < n_active; ++j) {
        if (drop[j]) continue;
        int w = (int)std::floor(vol[j] / cu.unit);
        if (w <= 0) {
            w = 1;
            cu.zero_weight[j] = 1;
            ++cu.n_zero_weight;
        }
        cu.weight[j] = w;
    }
    cu.units_total = cu.units + cu.n_zero_weight;
    return cu;
}

// Exact loads as integers scaled by 2^exponent.  Valid whenever the binary64
// volumes and capacity span a small enough exponent range (production data
// always does); otherwise the caller falls back to directed double intervals.
struct ScaledLoadDomain {
    bool ok = false;
    int exponent = 0;
    __uint128_t capacity = 0;
    std::vector<__uint128_t> weight;
};

inline bool binary64_to_scaled_uint128(double value, int exponent,
                                       __uint128_t& out) {
    if (value == 0.0) { out = 0; return true; }
    if (!(value > 0.0) || !std::isfinite(value)) return false;
    int e = 0;
    const double m = std::frexp(value, &e);          // value = m * 2^e
    const uint64_t sig = (uint64_t)std::ldexp(m, 53);  // exact integer
    const int shift = e - 53 - exponent;
    if (shift < 0) {
        if (-shift >= 64) return false;
        if ((sig & ((1ULL << -shift) - 1ULL)) != 0ULL) return false;
        out = (__uint128_t)(sig >> -shift);
        return true;
    }
    if (shift > 127 - 53) return false;
    out = ((__uint128_t)sig) << shift;
    return true;
}

inline ScaledLoadDomain build_scaled_load_domain(
    const double* vol, int n_active, int N, double capacity,
    const std::vector<char>& drop)
{
    ScaledLoadDomain d;
    int e_min = std::numeric_limits<int>::max();
    int e_top = std::numeric_limits<int>::min();
    auto note = [&](double v) {
        if (v == 0.0) return;
        const int top = std::ilogb(v);
        e_min = std::min(e_min, top - 52);
        e_top = std::max(e_top, top);
    };
    note(capacity);
    for (int j = 0; j < n_active; ++j) if (!drop[j]) note(vol[j]);
    d.weight.assign(N, 0);
    if (e_min == std::numeric_limits<int>::max()) {
        d.ok = true;   // every kept volume and the capacity are zero
        return d;
    }
    // A walk has at most 65535 customer visits (uint16 hop counter), so the
    // scaled sum stays below 2^(e_top+1+16-e_min) < 2^127.
    if ((long)e_top + 1 + 16 - (long)e_min > 120) return d;
    d.exponent = e_min;
    if (!binary64_to_scaled_uint128(capacity, e_min, d.capacity)) return d;
    for (int j = 0; j < n_active; ++j) {
        if (drop[j]) continue;
        if (!binary64_to_scaled_uint128(vol[j], e_min, d.weight[j])) return d;
    }
    d.ok = true;
    return d;
}

// Primal heuristic for the routed policy: nearest-neighbour seeds followed by
// first-improvement local search (removal, insertion, exchange, relocate,
// 2-opt) on ordinary-rounded reduced costs.  Loads are accumulated with
// upward rounding, so the returned route never exceeds the capacity; the
// caller re-evaluates its objective exactly.
inline std::vector<int> heuristic_route(
    const double* cost, const double* pi_alpha, const double* vol,
    int N, int n_active, int depot_start, int depot_end, double capacity,
    const std::vector<char>& drop)
{
    std::vector<double> red((size_t)N * N);
    for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
            red[(size_t)i * N + j] = cost[(size_t)i * N + j] - pi_alpha[j];
        }
    }
    auto R = [&](int i, int j) { return red[(size_t)i * N + j]; };
    auto route_value = [&](const std::vector<int>& r) {
        double v = 0.0;
        int prev = depot_start;
        for (int c : r) { v += R(prev, c); prev = c; }
        return v + R(prev, depot_end);
    };
    auto route_load_up = [&](const std::vector<int>& r) {
        double l = 0.0;
        for (int c : r) l = add_binary64_up(l, vol[c]);
        return l;
    };
    std::vector<int> kept;
    for (int j = 0; j < n_active; ++j) if (!drop[j]) kept.push_back(j);
    if (kept.empty()) return {};

    auto local_search = [&](std::vector<int> r) {
        std::vector<char> in_route(n_active, 0);
        for (int c : r) in_route[c] = 1;
        double value = route_value(r);
        double load = route_load_up(r);
        for (int pass = 0; pass < 500; ++pass) {
            bool improved = false;
            const int L = (int)r.size();
            // removal
            for (int p = 0; p < L && !improved; ++p) {
                const int a = p ? r[p - 1] : depot_start;
                const int b = (p + 1 < L) ? r[p + 1] : depot_end;
                const double delta = R(a, b) - R(a, r[p]) - R(r[p], b);
                if (delta < -1e-12) {
                    in_route[r[p]] = 0;
                    r.erase(r.begin() + p);
                    value += delta;
                    load = route_load_up(r);
                    improved = true;
                }
            }
            if (improved) continue;
            // insertion (best move)
            {
                double best_delta = -1e-12;
                int best_m = -1, best_p = -1;
                for (int m : kept) {
                    if (in_route[m]) continue;
                    if (add_binary64_up(load, vol[m]) > capacity) continue;
                    for (int p = 0; p <= L; ++p) {
                        const int a = p ? r[p - 1] : depot_start;
                        const int b = (p < L) ? r[p] : depot_end;
                        const double delta = R(a, m) + R(m, b) - R(a, b);
                        if (delta < best_delta) {
                            best_delta = delta;
                            best_m = m;
                            best_p = p;
                        }
                    }
                }
                if (best_m >= 0) {
                    r.insert(r.begin() + best_p, best_m);
                    in_route[best_m] = 1;
                    value += best_delta;
                    load = route_load_up(r);
                    continue;
                }
            }
            // exchange a routed customer for an unrouted one
            for (int p = 0; p < L && !improved; ++p) {
                const int a = p ? r[p - 1] : depot_start;
                const int b = (p + 1 < L) ? r[p + 1] : depot_end;
                const double base = R(a, r[p]) + R(r[p], b);
                double load_wo = 0.0;
                for (int t = 0; t < L; ++t) {
                    if (t != p) load_wo = add_binary64_up(load_wo, vol[r[t]]);
                }
                for (int m : kept) {
                    if (in_route[m]) continue;
                    if (add_binary64_up(load_wo, vol[m]) > capacity) continue;
                    const double delta = R(a, m) + R(m, b) - base;
                    if (delta < -1e-12) {
                        in_route[r[p]] = 0;
                        in_route[m] = 1;
                        r[p] = m;
                        value += delta;
                        load = route_load_up(r);
                        improved = true;
                        break;
                    }
                }
            }
            if (improved) continue;
            // relocate one customer inside the route
            for (int p = 0; p < L && !improved; ++p) {
                for (int q = 0; q < L && !improved; ++q) {
                    if (q == p) continue;
                    std::vector<int> trial = r;
                    const int c = trial[p];
                    trial.erase(trial.begin() + p);
                    trial.insert(trial.begin() + q, c);
                    const double tv = route_value(trial);
                    if (tv < value - 1e-12) {
                        r = std::move(trial);
                        value = tv;
                        improved = true;
                    }
                }
            }
            if (improved) continue;
            // 2-opt segment reversal (asymmetric-safe full re-evaluation)
            for (int p = 0; p + 1 < L && !improved; ++p) {
                for (int q = p + 1; q < L && !improved; ++q) {
                    std::reverse(r.begin() + p, r.begin() + q + 1);
                    const double tv = route_value(r);
                    if (tv < value - 1e-12) {
                        value = tv;
                        improved = true;
                    } else {
                        std::reverse(r.begin() + p, r.begin() + q + 1);
                    }
                }
            }
            if (!improved) break;
        }
        return r;
    };

    std::vector<int> best;
    double best_value = INF;
    auto consider = [&](const std::vector<int>& seed) {
        std::vector<int> r = local_search(seed);
        if (r.empty()) return;
        const double v = route_value(r);
        if (v < best_value) { best_value = v; best = std::move(r); }
    };
    // nearest-neighbour seed on reduced costs
    {
        std::vector<int> r;
        std::vector<char> used(n_active, 0);
        double load = 0.0;
        int cur = depot_start;
        while (true) {
            int best_j = -1;
            double best_arc = INF;
            for (int j : kept) {
                if (used[j]) continue;
                if (add_binary64_up(load, vol[j]) > capacity) continue;
                const double a = R(cur, j);
                if (a < best_arc) { best_arc = a; best_j = j; }
            }
            if (best_j < 0) break;
            if (best_arc >= 0.0 && cur != depot_start) break;
            used[best_j] = 1;
            load = add_binary64_up(load, vol[best_j]);
            cur = best_j;
            r.push_back(best_j);
        }
        if (!r.empty()) consider(r);
    }
    // single-customer seeds: the three best depot round trips
    {
        std::vector<std::pair<double, int>> single;
        for (int j : kept) {
            single.emplace_back(R(depot_start, j) + R(j, depot_end), j);
        }
        std::sort(single.begin(), single.end());
        for (size_t t = 0; t < single.size() && t < 3; ++t) {
            consider(std::vector<int>{single[t].second});
        }
    }
    return best;
}

template <typename Mask>
struct NgLabel {
    Mask mem;            // ng-memory: customers that may not be visited next
    Mask visited;        // exact visited set (a walk may repeat customers)
    __uint128_t load;    // exact scaled load (meaningful iff domain.ok)
    double cost_lb;      // directed-down prefix reduced cost
    double cost_ub;      // directed-up prefix reduced cost
    double cap_lb;       // directed-down prefix load
    double cap_ub;       // directed-up prefix load
    double key;          // certified full-objective lower bound of any completion
    uint32_t parent;
    uint16_t node;
    uint16_t hops;       // customer visits on the walk
    uint16_t depth;      // popcount(mem)
    bool alive;
};

template <typename Mask>
PctspResult solve_pctsp_ng_impl(
    const double* cost,
    const double* pi_alpha,
    const double* vol,
    int N, int n_active, int depot_start, int depot_end,
    double capacity,
    double pi_y,
    const double* vol_inactive,
    const double* pi_alpha_inactive,
    int n_inactive,
    double cutoff,
    uint64_t label_budget,
    double time_limit_s,
    int ng_size,
    int bound_ng_size,
    uint64_t bound_work_budget,
    int primal_top_k)
{
    using LabelT = NgLabel<Mask>;
    if (n_active > MAX_ACTIVE_BITS) {
        throw std::runtime_error(
            "pctsp: n_active>" + std::to_string(MAX_ACTIVE_BITS));
    }
    const int n_words = (n_active + 63) / 64;
    if (ng_size < 0) ng_size = 0;
    WallTimer timer(time_limit_s);
    PctspResult res{};
    res.ng_size_used = ng_size;
    double profile_mark_s = 0.0;

    // ---- inactive knapsack ----------------------------------------------
    KnapsackPrep knap_prep = prepare_knapsack(pi_alpha_inactive,
                                              vol_inactive, n_inactive);
    const double knap_ub_full = knapsack_ub(knap_prep, capacity);
    const KnapsackEnvelope knap_env(knap_prep);
    KnapsackDPTable knap_dp = build_knapsack_dp(knap_prep, capacity);
    KnapsackCache knap_cache;
    if (knap_dp.valid) {
        knap_cache = build_knapsack_cache(knap_prep, capacity, knap_dp.scale);
        knap_cache.has_dp = true;
    }
    res.knapsack_setup_time_s = timer.elapsed_s();

    // ---- directed reduced arcs -------------------------------------------
    std::vector<DirectedArcEndpoints> arc((size_t)N * N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const size_t a = (size_t)i * N + j;
            arc[a].lower = subtract_binary64_down(cost[a], pi_alpha[j]);
            arc[a].upper = subtract_binary64_up(cost[a], pi_alpha[j]);
        }
    }

    // ---- exact customer elimination + successor lists -------------------
    profile_mark_s = timer.elapsed_s();
    const std::vector<char> drop = detour_dominated_customers(
        cost, pi_alpha, vol, N, n_active, depot_start, depot_end,
        capacity, pi_y);
    int n_kept = 0;
    for (int j = 0; j < n_active; ++j) if (!drop[j]) ++n_kept;
    res.n_customers_dropped = n_active - n_kept;
    std::vector<std::vector<int>> successors(N);
    for (int i = 0; i < N; ++i) {
        if (i == depot_end) continue;
        auto& s = successors[i];
        s.reserve(n_kept + 1);
        for (int j = 0; j < n_active; ++j) {
            if (j != i && !drop[j]) s.push_back(j);
        }
        if (i != depot_start) s.push_back(depot_end);
    }
    res.arc_elimination_time_s = timer.elapsed_s() - profile_mark_s;

    // ---- exact load domain, ng-sets, completion bound -------------------
    const ScaledLoadDomain domain = build_scaled_load_domain(
        vol, n_active, N, capacity, drop);
    const NgSets<Mask> ng_sets = build_ng_sets<Mask>(
        cost, N, n_active, drop, ng_size);
    const std::vector<Mask>& ng = ng_sets.mask;
    profile_mark_s = timer.elapsed_s();

    // Completion bound.  B_j is built from the critical customers first (they
    // are remembered by every neighbourhood, hence never repeated by a DP
    // walk) and then the nearest members of N_j; B_j subset of N_j u critical
    // keeps the DP a relaxation of the labeling.  The table starts at the
    // cheapest precision level and is rebuilt (stronger level and/or larger
    // critical set) between DSSR iterations when the labeling is the more
    // expensive side.
    Mask critical;
    clear_visited(critical);
    std::vector<int> critical_order;     // DSSR criticals, most important first
    int bound_level = 0;
    CapacityUnits cu;
    CompletionBound cb;
    // weak_bound[j] = f(rows_max, j, empty memory): the loosest table entry
    // of j, used as a cache-resident pre-filter before the real lookup.
    std::vector<double> weak_bound(N, -INF);
    double bound_build_s = 0.0;          // wall time of the last build
    auto bound_member_lists = [&]() {
        std::vector<std::vector<int>> members(N);
        for (int j = 0; j < n_active; ++j) {
            if (drop[j]) continue;
            auto& m = members[j];
            for (int c : critical_order) if (c != j) m.push_back(c);
            const auto& ord = ng_sets.order[j];
            for (int t = 0; t < ng_size && t < (int)ord.size(); ++t) {
                const int g = ord[t];
                if (std::find(m.begin(), m.end(), g) == m.end()) m.push_back(g);
            }
        }
        return members;
    };
    auto build_bound = [&](int level) {
        const double t0 = timer.elapsed_s();
        const BoundLevel& bl = BOUND_LEVELS[level];
        cu = build_capacity_units(vol, N, n_active, capacity, drop,
                                  bl.cap_bits);
        cb = build_completion_bound(
            cu.weight, cu.units_total, arc, N, n_active, depot_start,
            depot_end, drop, bound_member_lists(),
            std::min(bl.ng_b, bound_ng_size), bound_work_budget);
        bound_level = level;
        for (int j = 0; j < n_active; ++j) {
            weak_bound[j] = drop[j] ? INF : cb.lookup(j, 0, cb.rows_max);
        }
        // Customers whose volume rounds to zero capacity units are always
        // elementary ("critical"): every other visit consumes >= one unit,
        // so a relaxed walk is finite (and the table may charge one unit per
        // zero-weight customer).  Making a customer critical never removes
        // an elementary route from the relaxation, so the set only grows.
        for (int j = 0; j < n_active; ++j) {
            if (!drop[j] && cu.zero_weight[j]) set_visited(critical, j);
        }
        bound_build_s = timer.elapsed_s() - t0;
        res.completion_lb_time_s += bound_build_s;
    };
    build_bound(0);
    // Estimated build cost of a level from the measured cost of the last one.
    auto estimate_build_s = [&](int level) -> double {
        const BoundLevel& bl = BOUND_LEVELS[level];
        const CapacityUnits cu_l = build_capacity_units(
            vol, N, n_active, capacity, drop, bl.cap_bits);
        const uint64_t rows = (uint64_t)cu_l.units_total + 1;
        const int ng_b = bound_ng_for_budget(
            rows, n_kept, std::min(bl.ng_b, bound_ng_size), bound_work_budget);
        const double work = (double)bound_transitions(rows, n_kept, ng_b);
        const double done = (double)std::max<uint64_t>(cb.transitions, 1);
        return bound_build_s * work / done;
    };

    // Capacity units a completion may still spend: floor(remaining/unit) plus
    // one unit per zero-weight customer (each visited at most once).
    auto remaining_units = [&](double cap_lb_used,
                               __uint128_t load) -> int {
        if (cu.units_total <= 0) return 0;
        __uint128_t q;
        if (domain.ok) {
            const __uint128_t rem = domain.capacity - load;  // load <= capacity
            const int shift = std::ilogb(cu.unit) - domain.exponent;
            if (shift >= 127) q = 0;
            else if (shift >= 0) q = rem >> shift;
            else q = rem << (-shift);   // < 512 by construction
        } else {
            // Power-of-two unit: the quotient is exact,
            // floor(up(R)/unit) >= floor(R/unit).
            const double rem_up = remaining_capacity_up(capacity, cap_lb_used);
            const double qd = std::floor(rem_up / cu.unit);
            q = (qd > 0.0) ? (__uint128_t)qd : 0;
        }
        q += (__uint128_t)cu.n_zero_weight;
        return (int)std::min<__uint128_t>(q, (__uint128_t)cu.units_total);
    };
    // ---- incumbents and thresholds --------------------------------------
    struct Candidate {
        double full_obj;
        ExactBinarySum exact_obj;
        int y;
        std::vector<int> path;
        KnapsackSol ksol;
    };
    std::vector<Candidate> candidates;
    const double user_cutoff = cutoff;
    // Exact best elementary (feasible) objective; y=0 has value 0.
    ExactBinarySum best_completed_exact;
    // Exact best objective over the current relaxation, elementary or not.
    ExactBinarySum best_relaxed_exact;
    bool relaxed_best_is_walk = false;
    std::vector<int> relaxed_best_path;
    // Upward-rounded pruning threshold: labels whose certified key exceeds it
    // cannot improve the current relaxation's optimum.
    double tau = 0.0;
    if (std::isfinite(user_cutoff) && user_cutoff < tau) tau = user_cutoff;
    auto refresh_tau = [&]() {
        tau = scaled_integer_to_binary64_up(best_relaxed_exact);
        if (std::isfinite(user_cutoff) && user_cutoff < tau) tau = user_cutoff;
    };
    {
        Candidate c0;
        c0.full_obj = 0.0;
        c0.exact_obj = {};
        c0.y = 0;
        c0.ksol.profit = 0.0;
        candidates.push_back(std::move(c0));
    }
    auto note_elementary = [&](const ExactBinarySum& exact_obj) {
        if (exact_obj < best_completed_exact) best_completed_exact = exact_obj;
        if (exact_obj <= best_relaxed_exact) {
            best_relaxed_exact = exact_obj;
            relaxed_best_is_walk = false;
            relaxed_best_path.clear();
        }
        refresh_tau();
    };
    auto knapsack_for_path = [&](const std::vector<int>& path,
                                        KnapsackSol& ksol) -> bool {
        const ExactBinarySum used = exact_path_active_volume(
            path, vol, n_active);
        const ExactBinarySum remaining = exact_remaining_capacity(
            capacity, used);
        if (remaining.negative) return false;
        const double rem_ub = scaled_integer_to_binary64_up(remaining);
        const bool exactly_binary64 = std::isfinite(rem_ub)
            && binary64_scaled_integer(rem_ub) == remaining;
        if (exactly_binary64 && knap_dp.valid && knap_cache.has_dp) {
            ksol = knapsack_solve_cached(knap_cache, knap_prep, rem_ub);
        } else {
            ksol = knapsack_solve_exact_capacity(knap_prep, rem_ub, remaining);
        }
        return true;
    };
    auto add_route_candidate = [&](std::vector<int> path) -> bool {
        KnapsackSol ksol;
        if (!knapsack_for_path(path, ksol)) return false;
        ExactBinarySum exact_obj = exact_pctsp_objective(
            1, path, ksol.selected, cost, pi_alpha, N,
            pi_alpha_inactive, pi_y);
        if (std::isfinite(user_cutoff)
            && exact_obj >= binary64_scaled_integer(user_cutoff)) {
            return false;
        }
        Candidate cand;
        cand.full_obj = scaled_integer_to_binary64_up(exact_obj);
        cand.y = 1;
        cand.path = std::move(path);
        cand.ksol = std::move(ksol);
        note_elementary(exact_obj);
        cand.exact_obj = std::move(exact_obj);
        candidates.push_back(std::move(cand));
        return true;
    };

    // yk=1, no routing, knapsack only (must assign at least one customer)
    {
        KnapsackSol ksol = knapsack_solve_nonempty(
            knap_prep, pi_alpha_inactive, vol_inactive, n_inactive, capacity);
        if (!ksol.selected.empty()) {
            const std::vector<int> empty_path;
            ExactBinarySum exact_obj = exact_pctsp_objective(
                1, empty_path, ksol.selected,
                cost, pi_alpha, N, pi_alpha_inactive, pi_y);
            if (!std::isfinite(user_cutoff)
                || exact_obj < binary64_scaled_integer(user_cutoff)) {
                Candidate ck;
                ck.full_obj = scaled_integer_to_binary64_up(exact_obj);
                ck.y = 1;
                ck.ksol = std::move(ksol);
                note_elementary(exact_obj);
                ck.exact_obj = std::move(exact_obj);
                candidates.push_back(std::move(ck));
            }
        }
    }
    // primal heuristic route
    profile_mark_s = timer.elapsed_s();
    {
        std::vector<int> customers = heuristic_route(
            cost, pi_alpha, vol, N, n_active, depot_start, depot_end,
            capacity, drop);
        if (!customers.empty()) {
            std::vector<int> path;
            path.reserve(customers.size() + 2);
            path.push_back(depot_start);
            path.insert(path.end(), customers.begin(), customers.end());
            path.push_back(depot_end);
            add_route_candidate(std::move(path));
        }
    }
    res.greedy_time_s = timer.elapsed_s() - profile_mark_s;
    const double preprocess_time_s = timer.elapsed_s();

    // ---- DSSR over ng-route relaxations ---------------------------------
    struct PQEntry {
        double priority;
        uint32_t lid;
        bool operator>(const PQEntry& o) const { return priority > o.priority; }
    };
    std::vector<LabelT> labels;
    std::vector<uint64_t> alive_count(N, 0);
    std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> work;
    struct Pending {
        int j;
        size_t arc_index;
        __uint128_t load;
        double cap_lb, cap_ub;
        double cost_lb, cost_ub;
        Mask mem;
        const float* entry;
    };
    std::vector<Pending> pending;
    pending.reserve(N);

    // Dominance archive.  Labels at j are grouped by their exact memory; a
    // group holds a 2-D Pareto frontier over (load, reduced cost), sorted by
    // cost_lb.  Because no record of a frontier dominates another, loads are
    // strictly decreasing along exact-cost order, so "is the new label
    // dominated" and "which records does it kill" are answered from a binary
    // search plus a short walk (interval overlaps are covered by the widest
    // cost interval in the group).  Groups are indexed by the ng-part of the
    // memory (a local bitmask over the first `dir_bits` members of N_j): a
    // label can only be dominated by groups in sub-slots and can only kill in
    // super-slots; the remaining bits are compared once per group.
    struct FrontierRec {
        __uint128_t load;
        double cost_lb, cost_ub;
        double cap_lb, cap_ub;
        uint32_t lid;
    };
    struct MemGroup {
        Mask mem;
        std::vector<FrontierRec> recs;   // sorted by cost_lb
        double max_width = 0.0;          // widest cost interval stored
    };
    struct NodeArchive {
        std::vector<std::vector<MemGroup>> slots;   // 1 << dir_bits entries
        SlotSet nonempty;                           // bit s: slots[s] non-empty
    };
    const int dir_bits = std::min(ng_size, SLOT_KEY_BITS);
    auto dir_key = [&](int j, const Mask& mem) -> uint32_t {
        uint32_t key = 0;
        const auto& ord = ng_sets.order[j];
        const int lim = std::min(dir_bits, (int)ord.size());
        for (int t = 0; t < lim; ++t) {
            if (is_visited(mem, ord[t])) key |= (1u << t);
        }
        return key;
    };
    std::vector<NodeArchive> archive(N);

    uint64_t n_labels_total = 0, n_pruned_dom = 0, n_pruned_lb = 0;
    uint64_t n_labels_popped = 0, n_arcs_considered = 0;
    uint64_t n_pruned_elementary = 0, n_pruned_capacity = 0;
    uint64_t n_dom_checks_forward = 0, n_dom_checks_reverse = 0;
    uint64_t max_bucket_size = 0, n_final_labels = 0;
    bool timed_out = false, label_budget_exhausted = false;
    bool interrupted = false, proved_optimal = false;
    double relaxation_lb = -INF;        // last completed iteration
    double partial_label_lb = INF;      // budget stop inside an expansion
    int iterations = 0;

    auto chain_path = [&](uint32_t lid, int tail) {
        std::vector<int> path;
        for (uint32_t cur = lid; cur != UINT32_MAX; cur = labels[cur].parent) {
            path.push_back((int)labels[cur].node);
        }
        std::reverse(path.begin(), path.end());
        if (tail >= 0) path.push_back(tail);
        return path;
    };

    double root_key = pctsp_full_lower_bound(cb.root_lb, knap_ub_full, pi_y);
    res.root_lb = root_key;
    // Table policy.  Between iterations: rebuild when the last relaxation
    // took longer than a rebuild (new criticals sharpen the table) and move
    // to the next precision level when it took longer than that build.
    // Inside an iteration: abort and escalate only when the relaxation runs
    // far past the estimated cost of the next level.  Incumbents and the
    // critical set stay valid across rebuilds.
    int n_escalations = 0;
    auto affordable = [&](double est) -> bool {
        return time_limit_s <= 0.0
            || est <= 0.5 * (time_limit_s - timer.elapsed_s());
    };
    auto escalation_threshold_s = [&]() -> double {
        if (bound_level + 1 >= N_BOUND_LEVELS) return INF;
        const double est = estimate_build_s(bound_level + 1);
        if (!affordable(est)) return INF;
        return std::max(2.0 * est, 0.02);
    };
    auto rebuild_after_iteration = [&](double last_iteration_s,
                                       bool criticals_grew) {
        int level = bound_level;
        if (level + 1 < N_BOUND_LEVELS) {
            const double est = estimate_build_s(level + 1);
            if (last_iteration_s > 0.5 * est && affordable(est)) ++level;
        }
        bool rebuild = level != bound_level;
        if (!rebuild && criticals_grew) {
            const double est = estimate_build_s(level);
            rebuild = last_iteration_s > est && affordable(est);
        }
        if (!rebuild) return;
        if (level != bound_level) ++n_escalations;
        build_bound(level);
        root_key = std::max(root_key, pctsp_full_lower_bound(
            cb.root_lb, knap_ub_full, pi_y));
        res.root_lb = root_key;
    };
    double escalate_after_s = escalation_threshold_s();
    const int max_iterations = n_kept + 1 + N_BOUND_LEVELS;
    while (true) {
        // The root key lower-bounds every routed policy.  Once it exceeds the
        // best elementary incumbent no route can improve on it.
        double incumbent_up = scaled_integer_to_binary64_up(best_completed_exact);
        if (std::isfinite(user_cutoff) && user_cutoff < incumbent_up) {
            incumbent_up = user_cutoff;
        }
        if (n_kept == 0 || root_key > incumbent_up + EPS_LB_PRUNE) {
            proved_optimal = true;
            if (n_kept > 0) relaxation_lb = std::max(relaxation_lb, root_key);
            break;
        }
        ++iterations;
        const double iteration_started_s = timer.elapsed_s();
        bool escalate = false;
        labels.clear();
        labels.reserve(1 << 14);
        for (auto& a : archive) {
            a.slots.assign((size_t)1 << dir_bits, {});
            a.nonempty = SlotSet{};
        }
        std::fill(alive_count.begin(), alive_count.end(), 0);
        work = decltype(work)();
        // The previous relaxed optimum is infeasible for the tighter
        // relaxation; elementary candidates remain feasible.
        best_relaxed_exact = best_completed_exact;
        relaxed_best_is_walk = false;
        relaxed_best_path.clear();
        refresh_tau();
        {
            LabelT root;
            clear_visited(root.mem);
            clear_visited(root.visited);
            root.load = 0;
            root.cost_lb = root.cost_ub = 0.0;
            root.cap_lb = root.cap_ub = 0.0;
            root.key = root_key;
            root.parent = UINT32_MAX;
            root.node = (uint16_t)depot_start;
            root.hops = 0;
            root.depth = 0;
            root.alive = true;
            labels.push_back(root);
            ++n_labels_total;
            work.push({root_key, 0});
        }
        bool iteration_complete = false;
    while (!work.empty()) {
        if (timer.expired()) {
            timed_out = true;
                interrupted = true;
            break;
        }
            if ((n_labels_popped & 255u) == 0u
                && timer.elapsed_s() - iteration_started_s > escalate_after_s) {
                escalate = true;
                break;
            }
            const PQEntry top = work.top();
        work.pop();
            if (!labels[top.lid].alive) continue;
        ++n_labels_popped;
            if (top.priority > tau + EPS_LB_PRUNE) {
                // Every pending key is a certified bound >= this one.
                iteration_complete = true;
                break;
            }
            const uint32_t lid = top.lid;
            const LabelT L = labels[lid];
            const auto& succ = successors[L.node];
            // Inactive profit bound before adding any successor: at least as
            // large as after, so keys built from it stay valid lower bounds.
            const double label_inactive_ub = knap_env.ub(
                remaining_capacity_up(capacity, L.cap_lb));

            // Pass 1: cheap feasibility / weak-bound filters; survivors get
            // their completion-table entry prefetched so pass 2 does not
            // stall on one cache miss per arc.
            pending.clear();
            for (size_t pos = 0; pos < succ.size(); ++pos) {
                ++n_arcs_considered;
                const int j = succ[pos];
                const bool to_depot = (j == depot_end);
                Pending P;
                P.j = j;
                P.load = L.load;
                P.cap_lb = L.cap_lb;
                P.cap_ub = L.cap_ub;
                if (!to_depot) {
                    if (is_visited(L.mem, j)) {
                        ++n_pruned_elementary;
            continue;
        }
                    if (L.hops >= 65000) continue;  // unreachable, see critical rule
                    if (domain.ok) {
                        P.load += domain.weight[j];
                        if (P.load > domain.capacity) {
                            ++n_pruned_capacity;
                            continue;
                        }
                    }
                    P.cap_lb = add_binary64_down(P.cap_lb, vol[j]);
                    P.cap_ub = add_binary64_up(P.cap_ub, vol[j]);
                    if (!domain.ok && P.cap_lb > capacity) {
                        ++n_pruned_capacity;
                        continue;
                    }
                }
                const size_t arc_index = (size_t)L.node * N + j;
                P.arc_index = arc_index;
                P.cost_lb = add_binary64_down(L.cost_lb, arc[arc_index].lower);
                P.cost_ub = add_binary64_up(L.cost_ub, arc[arc_index].upper);
                clear_visited(P.mem);
                P.entry = nullptr;
                if (!to_depot) {
                    const double weak_key = pctsp_full_lower_bound(
                        add_binary64_down(P.cost_lb, weak_bound[j]),
                        label_inactive_ub, pi_y);
                    if (weak_key > tau + EPS_LB_PRUNE) {
                ++n_pruned_lb;
                continue;
            }
                    // ng-memory: keep what N_j remembers plus every critical
                    // customer, then remember j itself.
                    P.mem = mask_or(mask_and(L.mem, ng[j]),
                                    mask_and(L.mem, critical));
                    set_visited(P.mem, j);
                    P.entry = cb.entry(j, cb.local_index(j, P.mem),
                                       remaining_units(P.cap_lb, P.load));
                    __builtin_prefetch(P.entry);
                }
                pending.push_back(P);
            }

            // Pass 2: certified key, completion of walks, dominance, insertion.
            for (size_t pos = 0; pos < pending.size(); ++pos) {
                const Pending& P = pending[pos];
                const int j = P.j;
                const bool to_depot = (j == depot_end);
                const __uint128_t new_load = P.load;
                const double new_cap_lb = P.cap_lb, new_cap_ub = P.cap_ub;
                const size_t arc_index = P.arc_index;
                const double new_cost_lb = P.cost_lb;
                const double new_cost_ub = P.cost_ub;
                const Mask& new_mem = P.mem;
                const double completion = to_depot ? 0.0 : (double)*P.entry;
                if (completion >= INF) continue;
                const double inactive_ub = knap_env.ub(
                    remaining_capacity_up(capacity, new_cap_lb));
                const double key = pctsp_full_lower_bound(
                    add_binary64_down(new_cost_lb, completion),
                    inactive_ub, pi_y);
                if (key > tau + EPS_LB_PRUNE) {
                    ++n_pruned_lb;
                    continue;
                }
                if (to_depot) {
                    // Evaluate the completed walk exactly.
                    ++n_final_labels;
                    std::vector<int> path = chain_path(lid, depot_end);
                    KnapsackSol ksol;
                    if (!knapsack_for_path(path, ksol)) continue;
                    ExactBinarySum exact_obj = exact_pctsp_objective(
                        1, path, ksol.selected, cost, pi_alpha, N,
                        pi_alpha_inactive, pi_y);
                    if (std::isfinite(user_cutoff)
                        && exact_obj >= binary64_scaled_integer(user_cutoff)) {
                        continue;
                    }
                    const bool elementary =
                        (int)L.hops == mask_popcount(L.visited);
                    if (elementary) {
                        Candidate cand;
                        cand.full_obj = scaled_integer_to_binary64_up(exact_obj);
                        cand.y = 1;
                        cand.path = std::move(path);
                        cand.ksol = std::move(ksol);
                        note_elementary(exact_obj);
                        cand.exact_obj = std::move(exact_obj);
                        candidates.push_back(std::move(cand));
                    } else if (exact_obj < best_relaxed_exact) {
                        best_relaxed_exact = std::move(exact_obj);
                        relaxed_best_is_walk = true;
                        relaxed_best_path = std::move(path);
                        refresh_tau();
                    }
                continue;
            }

                Mask new_visited = L.visited;
                set_visited(new_visited, j);
                const uint16_t new_depth = (uint16_t)mask_popcount(new_mem);
                const uint32_t new_key = dir_key(j, new_mem);
                NodeArchive& arch = archive[j];
                if (alive_count[j] > max_bucket_size) max_bucket_size = alive_count[j];

                // Exact comparisons: directed intervals first, exact sums on overlap.
                bool new_exact_ready = false;
            ExactBinarySum new_exact_cost;
            auto exact_new_cost = [&]() -> const ExactBinarySum& {
                    if (!new_exact_ready) {
                    new_exact_cost = exact_label_reduced_cost(
                        labels, lid, cost, pi_alpha, N);
                        add_binary64_exact(new_exact_cost, cost[arc_index]);
                        subtract_binary64_exact(new_exact_cost, pi_alpha[j]);
                        new_exact_ready = true;
                }
                return new_exact_cost;
            };
                // -1: existing < new, 0: equal, +1: existing > new
                auto compare_cost = [&](const FrontierRec& E) -> int {
                    if (E.cost_ub < new_cost_lb) return -1;
                    if (E.cost_lb > new_cost_ub) return 1;
                    const ExactBinarySum existing = exact_label_reduced_cost(
                        labels, E.lid, cost, pi_alpha, N);
                    const ExactBinarySum& candidate = exact_new_cost();
                    if (existing < candidate) return -1;
                    if (existing > candidate) return 1;
                return 0;
            };
                // -1/0/+1 as above, +2 when undecidable (fallback intervals overlap)
                auto compare_load = [&](const FrontierRec& E) -> int {
                    if (domain.ok) {
                        if (E.load < new_load) return -1;
                        if (E.load > new_load) return 1;
                        return 0;
                    }
                    if (E.cap_ub < new_cap_lb) return -1;
                    if (E.cap_lb > new_cap_ub) return 1;
                    if (E.cap_lb == E.cap_ub && new_cap_lb == new_cap_ub
                        && E.cap_lb == new_cap_lb) return 0;
                    return 2;
                };

                // Forward: is the new label dominated by a record of a group
                // whose memory is a subset of new_mem?  Walk the frontier
                // backwards from the last record that may cost <= new.  The
                // first record with a larger load is a stopper: everything
                // exactly cheaper than it carries an even larger load.
            bool dominated = false;
                auto scan_forward = [&](const MemGroup& G) {
                    const auto& F = G.recs;
                    size_t idx = std::upper_bound(
                        F.begin(), F.end(), new_cost_ub,
                        [](double v, const FrontierRec& E) { return v < E.cost_lb; })
                        - F.begin();
                    bool stopper = false;
                    double stop_lb = 0.0;
                    while (idx-- > 0) {
                        const FrontierRec& E = F[idx];
                        if (stopper && E.cost_lb < stop_lb - G.max_width) break;
                            ++n_dom_checks_forward;
                        const int lc = compare_load(E);
                        if (lc == -1 || lc == 0) {
                            if (compare_cost(E) <= 0) { dominated = true; return; }
                        } else if (lc == 1 && !stopper) {
                            stopper = true;
                            stop_lb = E.cost_lb;
                        }
                    }
                };
                auto scan_forward_slot = [&](const std::vector<MemGroup>& slot) {
                    for (const MemGroup& G : slot) {
                        if (!subset_of(G.mem, new_mem, n_words)) continue;
                        scan_forward(G);
                        if (dominated) return;
                    }
                };
                // Non-empty slots whose key is a subset of new_key.
                for (int w = 0; w < SLOT_WORDS && !dominated; ++w) {
                    uint64_t bits = SLOT_SUBSETS[new_key].w[w] & arch.nonempty.w[w];
                    while (bits != 0ULL) {
                        const uint32_t s = (uint32_t)(w * 64 + __builtin_ctzll(bits));
                        bits &= bits - 1ULL;
                        scan_forward_slot(arch.slots[s]);
                        if (dominated) break;
                }
            }
            if (dominated) { ++n_pruned_dom; continue; }

                LabelT NL;
                NL.mem = new_mem;
                NL.visited = new_visited;
                NL.load = new_load;
                NL.cost_lb = new_cost_lb;
                NL.cost_ub = new_cost_ub;
                NL.cap_lb = new_cap_lb;
                NL.cap_ub = new_cap_ub;
                NL.key = key;
            NL.parent = lid;
                NL.node = (uint16_t)j;
                NL.hops = (uint16_t)(L.hops + 1);
                NL.depth = new_depth;
            NL.alive = true;
                const uint32_t new_id = (uint32_t)labels.size();
            labels.push_back(NL);
                ++n_labels_total;

                // Reverse: kill records the new label dominates in groups whose
                // memory contains new_mem.  Candidates cost at least new_cost;
                // the first record with a smaller load is a stopper: everything
                // exactly dearer than it carries an even smaller load.
                auto scan_reverse = [&](MemGroup& G) {
                    auto& F = G.recs;
                    const double from = new_cost_lb - G.max_width;
                    size_t k = std::lower_bound(
                        F.begin(), F.end(), from,
                        [](const FrontierRec& E, double v) { return E.cost_lb < v; })
                        - F.begin();
                    size_t write = k;
                    bool stopper = false;
                    double stop_lb = 0.0;
                    for (; k < F.size(); ++k) {
                        const FrontierRec& E = F[k];
                        if (stopper && E.cost_lb > stop_lb + G.max_width) {
                            if (write == k) return;   // nothing killed: done
                            break;
                        }
                            ++n_dom_checks_reverse;
                        bool kill = false;
                        const int lc = compare_load(E);
                        if (lc == 0 || lc == 1) {
                            // A full tie inside the same group would have
                            // dominated the new label in the forward scan.
                            kill = compare_cost(E) >= 0;
                        } else if (lc == -1 && !stopper) {
                            stopper = true;
                            stop_lb = E.cost_lb;
                        }
                        if (kill) {
                            labels[E.lid].alive = false;
                            --alive_count[j];
                                ++n_pruned_dom;
                        } else {
                            if (write != k) F[write] = F[k];
                            ++write;
                        }
                    }
                    for (; k < F.size(); ++k) F[write++] = F[k];
                    F.resize(write);
                };
                auto scan_reverse_slot = [&](std::vector<MemGroup>& slot) {
                    for (MemGroup& G : slot) {
                        if (!G.recs.empty() && subset_of(new_mem, G.mem, n_words)) {
                            scan_reverse(G);
                        }
                    }
                };
                // Non-empty slots whose key is a superset of new_key.
                for (int w = 0; w < SLOT_WORDS; ++w) {
                    uint64_t bits = SLOT_SUPERSETS[new_key].w[w] & arch.nonempty.w[w];
                    while (bits != 0ULL) {
                        const uint32_t s = (uint32_t)(w * 64 + __builtin_ctzll(bits));
                        bits &= bits - 1ULL;
                        scan_reverse_slot(arch.slots[s]);
                    }
                }
                {
                    FrontierRec rec;
                    rec.load = new_load;
                    rec.cost_lb = new_cost_lb;
                    rec.cost_ub = new_cost_ub;
                    rec.cap_lb = new_cap_lb;
                    rec.cap_ub = new_cap_ub;
                    rec.lid = new_id;
                    auto& slot = arch.slots[new_key];
                    if (slot.empty()) {
                        arch.nonempty.w[new_key >> 6] |= 1ULL << (new_key & 63u);
                    }
                    MemGroup* G = nullptr;
                    for (MemGroup& g : slot) {
                        if (equals_mask(g.mem, new_mem, n_words)) { G = &g; break; }
                    }
                    if (G == nullptr) {
                        slot.emplace_back();
                        G = &slot.back();
                        G->mem = new_mem;
                    }
                    auto& F = G->recs;
                    const auto pos_it = std::upper_bound(
                        F.begin(), F.end(), new_cost_lb,
                        [](double v, const FrontierRec& E) { return v < E.cost_lb; });
                    F.insert(pos_it, rec);
                    G->max_width = std::max(G->max_width,
                                            new_cost_ub - new_cost_lb);
                }
                ++alive_count[j];
                if (alive_count[j] > max_bucket_size) max_bucket_size = alive_count[j];
                work.push({key, new_id});

            if (label_budget && labels.size() >= label_budget) {
                label_budget_exhausted = true;
                    interrupted = true;
                    // Unprocessed successors of L still bound the lost branches.
                    for (size_t rest = pos + 1; rest < pending.size(); ++rest) {
                        const Pending& R = pending[rest];
                        const double r_completion = (R.j == depot_end)
                            ? 0.0 : (double)*R.entry;
                        if (r_completion >= INF) continue;
                        const double r_key = pctsp_full_lower_bound(
                            add_binary64_down(R.cost_lb, r_completion),
                            knap_env.ub(remaining_capacity_up(capacity, R.cap_lb)),
                            pi_y);
                        partial_label_lb = std::min(partial_label_lb, r_key);
                    }
                break;
            }
        }
            if (interrupted) break;
        }
        if (interrupted) break;
        const double iteration_s = timer.elapsed_s() - iteration_started_s;
        if (escalate) {
            build_bound(bound_level + 1);
            ++n_escalations;
            root_key = std::max(root_key, pctsp_full_lower_bound(
                cb.root_lb, knap_ub_full, pi_y));
            res.root_lb = root_key;
            escalate_after_s = escalation_threshold_s();
            continue;
        }
        (void)iteration_complete;
        // Iteration solved: best_relaxed_exact is a certified lower bound on
        // every elementary policy (see the certifying argument above).
        const double relaxed_down =
            scaled_integer_to_binary64_down(best_relaxed_exact);
        relaxation_lb = std::max(relaxation_lb, relaxed_down);
        if (!relaxed_best_is_walk) {
            proved_optimal = true;
            break;
        }
        // DSSR: customers repeated by the relaxed optimum become critical.
        {
            std::vector<int> count(n_active, 0);
            for (int node : relaxed_best_path) {
                if (node >= 0 && node < n_active) ++count[node];
            }
            bool grew = false;
            for (int j = 0; j < n_active; ++j) {
                if (count[j] > 1 && !is_visited(critical, j)) {
                    set_visited(critical, j);
                    critical_order.push_back(j);
                    grew = true;
                }
            }
            if (!grew || iterations >= max_iterations) {
                // Cannot happen (a repeated customer is never critical); keep
                // the certified bound rather than looping forever.
                interrupted = true;
                break;
            }
            rebuild_after_iteration(iteration_s, true);
            escalate_after_s = escalation_threshold_s();
        }
    }
    // ---- certified global lower bound ------------------------------------
    const double frontier_started_s = timer.elapsed_s();
    double frontier_lb = INF;
    uint64_t n_pending_labels = 0;
    if (interrupted) {
        frontier_lb = scaled_integer_to_binary64_down(best_relaxed_exact);
        auto work_copy = work;
        while (!work_copy.empty()) {
            const PQEntry entry = work_copy.top();
            work_copy.pop();
            if (!labels[entry.lid].alive) continue;
            ++n_pending_labels;
            frontier_lb = std::min(frontier_lb, entry.priority);
        }
        frontier_lb = std::min(frontier_lb, partial_label_lb);
    }
    double completed_or_cutoff_lb =
        scaled_integer_to_binary64_down(best_completed_exact);
    if (std::isfinite(user_cutoff)) {
        completed_or_cutoff_lb = std::min(completed_or_cutoff_lb, user_cutoff);
    }
    const double res_lb_global = std::min(
        completed_or_cutoff_lb, std::max(relaxation_lb, frontier_lb));
    res.frontier_lb_time_s = timer.elapsed_s() - frontier_started_s;
    const double label_time_s = timer.elapsed_s() - preprocess_time_s;

    // ---- post-processing ----------------------------------------------
    const double postprocess_started_s = timer.elapsed_s();
    // Tie-break for Lagrangian subgradient informativeness (unchanged):
    // exact objective, then y=1, then more routed customers, then a larger
    // inactive selection.
    std::stable_sort(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) {
            if (a.exact_obj != b.exact_obj) return a.exact_obj < b.exact_obj;
            if (a.y != b.y) return a.y > b.y;
            const int a_vis = a.path.empty() ? 0 : (int)a.path.size() - 2;
            const int b_vis = b.path.empty() ? 0 : (int)b.path.size() - 2;
            if (a_vis != b_vis) return a_vis > b_vis;
            return a.ksol.selected.size() > b.ksol.selected.size();
        });
    res.postprocess_time_s = timer.elapsed_s() - postprocess_started_s;

    res.n_labels = n_labels_total;
    res.n_pruned_dom = n_pruned_dom;
    res.n_pruned_lb = n_pruned_lb;
    res.timed_out = timed_out;
    res.label_budget_exhausted = label_budget_exhausted;
    res.preprocess_time_s = preprocess_time_s;
    res.label_time_s = label_time_s;
    res.n_labels_popped = n_labels_popped;
    res.n_arcs_considered = n_arcs_considered;
    res.n_pruned_arc = 0;
    res.n_pruned_elementary = n_pruned_elementary;
    res.n_pruned_capacity = n_pruned_capacity;
    res.n_dom_checks_forward = n_dom_checks_forward;
    res.n_dom_checks_reverse = n_dom_checks_reverse;
    res.max_bucket_size = max_bucket_size;
    res.n_final_labels = n_final_labels;
    res.n_pending_labels = n_pending_labels;
    res.dssr_iterations = iterations;
    res.relaxation_lb = relaxation_lb;
    res.bound_ng_size = cb.ng_b;
    res.bound_level = bound_level;
    res.n_escalations = n_escalations;

    const Candidate& best = candidates.front();
    res.obj_val = best.full_obj;
    res.y = best.y;
    res.path = best.path;
    res.alpha_inactive.assign(n_inactive, 0);
    if (best.y == 1) {
        for (int idx : best.ksol.selected) res.alpha_inactive[idx] = 1;
    }
    const bool cutoff_unresolved = std::isfinite(user_cutoff)
        && !(best.exact_obj < binary64_scaled_integer(user_cutoff));
    res.cutoff_bound_triggered = cutoff_unresolved;
    // An interrupted run whose certified bound already reaches the incumbent
    // has proved the incumbent optimal (the heuristic route is often exact).
    const double incumbent_down = scaled_integer_to_binary64_down(best.exact_obj);
    const bool closed_by_bound = !cutoff_unresolved
        && std::max(relaxation_lb, frontier_lb) >= incumbent_down;
    const bool proved = proved_optimal || (interrupted && closed_by_bound);
    res.status = (!proved || cutoff_unresolved) ? 2 : 0;
    res.lb = res.status == 0 ? incumbent_down : res_lb_global;

    // Export only already stored elementary primal witnesses, after the
    // unchanged best solution / status / LB have been finalized. The core
    // searches for ONE optimum, so these are not a complete top-K ranking.
    // Bound both returned payload and additional duplicate-scan work.
    if (primal_top_k > 1) {
        constexpr size_t MAX_PRIMAL_EXPORT_SCAN = 4096;
        std::vector<VisitedMask> exported_sets;
        exported_sets.reserve((size_t)primal_top_k);
        res.candidate_routes.reserve((size_t)primal_top_k);
        size_t examined = 0;
        for (const Candidate& cand : candidates) {
            if (examined++ >= MAX_PRIMAL_EXPORT_SCAN
                || res.candidate_routes.size() >= (size_t)primal_top_k) break;
            // Exclude idle and knapsack-only states: neither is a route column.
            if (cand.y != 1 || cand.path.size() < 3
                || cand.path.front() != depot_start
                || cand.path.back() != depot_end) continue;
            VisitedMask customers{};
            bool elementary = true;
            for (size_t pos = 1; pos + 1 < cand.path.size(); ++pos) {
                const int j = cand.path[pos];
                if (j < 0 || j >= n_active || is_visited(customers, j)) {
                    elementary = false;
                    break;
                }
                set_visited(customers, j);
            }
            if (!elementary) continue;
            bool duplicate = false;
            for (const VisitedMask& prior : exported_sets) {
                if (equals_mask(customers, prior, n_words)) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) continue;
            // Current candidate sorting uses exact full objective. For one
            // active customer set the prize and exact remaining capacity are
            // fixed, hence its inactive knapsack and reward are also fixed:
            // the first saved candidate has minimum original route cost for
            // that set. In the LRP domain there are no inactive selections.
            PctspPrimalRoute exported;
            exported.obj_val = cand.full_obj;
            exported.y = cand.y;
            exported.path = cand.path;
            exported.alpha_inactive.assign(n_inactive, 0);
            for (int j : cand.ksol.selected) exported.alpha_inactive[j] = 1;
            exported_sets.push_back(customers);
            res.candidate_routes.push_back(std::move(exported));
        }
    }
    res.total_time_s = timer.elapsed_s();
    return res;
}

PctspResult solve_pctsp_core(
    const double* cost,
    const double* pi_alpha,
    const double* vol,
    int N, int n_active, int depot_start, int depot_end,
    double capacity,
    double pi_y,
    const double* vol_inactive,
    const double* pi_alpha_inactive,
    int n_inactive,
    double cutoff,
    int top_k,
    uint64_t label_budget,
    double time_limit_s = 0.0,
    int ng_size = 8,
    int bound_ng_size = MAX_BOUND_NG,
    uint64_t bound_work_budget = 200000000ULL
) {
    // Preserve legacy top_k<=1 behavior; optional output is bounded at 64.
    const int primal_top_k = std::max(1, std::min(top_k, 64));
    if (n_active <= 64) {
        return solve_pctsp_ng_impl<uint64_t>(
            cost, pi_alpha, vol,
            N, n_active, depot_start, depot_end,
            capacity, pi_y,
            vol_inactive, pi_alpha_inactive, n_inactive,
            cutoff, label_budget, time_limit_s, ng_size,
            bound_ng_size, bound_work_budget, primal_top_k);
    }
    return solve_pctsp_ng_impl<VisitedMask>(
        cost, pi_alpha, vol,
        N, n_active, depot_start, depot_end,
        capacity, pi_y,
        vol_inactive, pi_alpha_inactive, n_inactive,
        cutoff, label_budget, time_limit_s, ng_size,
        bound_ng_size, bound_work_budget, primal_top_k);
}

// ============================================================================
// Python wrapper for solve_pctsp
// ============================================================================
py::dict solve_pctsp_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> cost_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> pi_alpha_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> vol_arr,
    int n_active,
    int depot_start,
    int depot_end,
    double capacity,
    double pi_y,
    py::array_t<double, py::array::c_style | py::array::forcecast> vol_inactive_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> pi_alpha_inactive_arr,
    double cutoff,
    int top_k,
    uint64_t label_budget,
    bool use_bidirectional,
    double time_limit_s,
    double bi_probe_time_s,
    int ng_size,
    int bound_ng_size,
    uint64_t bound_work_budget
) {
    auto c_buf = cost_arr.request();
    auto pi_buf = pi_alpha_arr.request();
    auto vol_buf = vol_arr.request();
    auto vi_buf = vol_inactive_arr.request();
    auto pia_buf = pi_alpha_inactive_arr.request();
    if (ng_size < 0 || ng_size > MAX_ACTIVE_BITS) {
        throw std::runtime_error("ng_size must be in [0, 256]");
    }
    if (bound_ng_size < 0 || bound_ng_size > MAX_BOUND_NG) {
        throw std::runtime_error("bound_ng_size must be in [0, 8]");
    }
    if (bound_ng_size > ng_size) bound_ng_size = ng_size;
    if (c_buf.ndim != 2 || c_buf.shape[0] != c_buf.shape[1]) {
        throw std::runtime_error("cost must be (N,N)");
    }
    int N = (int)c_buf.shape[0];
    if (pi_buf.ndim != 1 || vol_buf.ndim != 1
        || (int)pi_buf.shape[0] != N || (int)vol_buf.shape[0] != N) {
        throw std::runtime_error("pi_alpha / vol length must equal N");
    }
    if (depot_start < 0 || depot_start >= N || depot_end < 0 || depot_end >= N
        || depot_start == depot_end) {
        throw std::runtime_error("depot indices invalid");
    }
    if (n_active < 0 || n_active > N - 2) {
        throw std::runtime_error("n_active invalid");
    }
    if (N != n_active + 2
        || depot_start < n_active || depot_end < n_active) {
        throw std::runtime_error(
            "PCTSP layout must be active customers followed by two depots");
    }
    if (vi_buf.ndim != 1 || pia_buf.ndim != 1) {
        throw std::runtime_error(
            "vol_inactive / pi_alpha_inactive must be one-dimensional");
    }
    int n_inactive = (int)vi_buf.shape[0];
    if ((int)pia_buf.shape[0] != n_inactive) {
        throw std::runtime_error("vol_inactive / pi_alpha_inactive length mismatch");
    }
    if (!std::isfinite(capacity) || capacity < 0.0) {
        throw std::runtime_error("capacity must be finite and nonnegative");
    }
    if (!std::isfinite(pi_y)) {
        throw std::runtime_error("pi_y must be finite");
    }
    validate_cutoff(cutoff);
    if (!std::isfinite(time_limit_s) || time_limit_s < 0.0) {
        throw std::runtime_error("time_limit_s must be finite and nonnegative");
    }
    // Retired bidirectional solver: both switches are accepted so existing
    // call sites keep working, but every call runs the certifying core.
    (void)use_bidirectional;
    (void)bi_probe_time_s;
    validate_finite_array(c_buf, "cost");
    validate_finite_array(pi_buf, "pi_alpha");
    validate_nonnegative_array(vol_buf, "vol");
    validate_nonnegative_array(vi_buf, "vol_inactive");
    validate_finite_array(pia_buf, "pi_alpha_inactive");

    PctspResult r{};
    {
        py::gil_scoped_release nogil;
        WallTimer wrapper_timer(0.0);
        const double* active_volume = (const double*)vol_buf.ptr;
        const double* inactive_volume = (const double*)vi_buf.ptr;
        bool some_customer_fits = false;
        for (int j = 0; j < n_active && !some_customer_fits; ++j) {
            some_customer_fits = active_volume[j] <= capacity;
        }
        for (int j = 0; j < n_inactive && !some_customer_fits; ++j) {
            some_customer_fits = inactive_volume[j] <= capacity;
        }

        if (!some_customer_fits) {
            // Stage 3 enforces sum(alpha) >= y.  If no customer can fit, the
            // only feasible policy is therefore y=0, alpha=0, with value 0.
            // Prove this before the O(n^3) completion-bound preprocessing;
            // this is exact for both forward and backward uses.
            r.obj_val = 0.0;
            r.lb = 0.0;
            r.y = 0;
            r.path.clear();
            r.alpha_inactive.assign(n_inactive, 0);
            r.n_labels = 1;
            const bool cutoff_unresolved = std::isfinite(cutoff)
                && !(0.0 < cutoff);
            r.cutoff_bound_triggered = cutoff_unresolved;
            r.status = cutoff_unresolved ? 2 : 0;
            r.preprocess_time_s = wrapper_timer.elapsed_s();
            r.total_time_s = r.preprocess_time_s;
            } else {
                    r = solve_pctsp_core(
                        (const double*)c_buf.ptr,
                        (const double*)pi_buf.ptr,
                        (const double*)vol_buf.ptr,
                        N, n_active, depot_start, depot_end,
                        capacity, pi_y,
                        (const double*)vi_buf.ptr,
                        (const double*)pia_buf.ptr,
                        n_inactive,
                        std::isfinite(cutoff) ? cutoff : INF,
                top_k, label_budget, time_limit_s, ng_size,
                bound_ng_size, bound_work_budget);
        }
    }

    py::list path_list;
    for (int n : r.path) path_list.append(n);
    py::list alpha_list;
    for (int a : r.alpha_inactive) alpha_list.append(a);

    py::dict out;
    out["obj_val"] = r.obj_val;
    out["lb"] = r.lb;          // directed LB; status==0 may differ from UB by 1 ulp
    out["y"] = r.y;
    out["path"] = path_list;
    out["alpha_inactive"] = alpha_list;
    if (top_k > 1) {
        py::list exported_routes;
        for (const PctspPrimalRoute& candidate : r.candidate_routes) {
            py::dict witness;
            witness["obj_val"] = candidate.obj_val;
            witness["y"] = candidate.y;
            witness["path"] = py::cast(candidate.path);
            witness["alpha_inactive"] = py::cast(candidate.alpha_inactive);
            exported_routes.append(std::move(witness));
        }
        out["candidate_routes"] = std::move(exported_routes);
    }
    out["status"] = r.status;
    out["n_labels"] = r.n_labels;
    out["n_pruned_dom"] = r.n_pruned_dom;
    out["n_pruned_lb"] = r.n_pruned_lb;
    out["timed_out"] = r.timed_out;
    out["label_budget_exhausted"] = r.label_budget_exhausted;
    out["cutoff_bound_triggered"] = r.cutoff_bound_triggered;
    out["preprocess_time_s"] = r.preprocess_time_s;
    out["label_time_s"] = r.label_time_s;
    out["total_time_s"] = r.total_time_s;
    out["completion_lb_time_s"] = r.completion_lb_time_s;
    out["knapsack_setup_time_s"] = r.knapsack_setup_time_s;
    out["arc_elimination_time_s"] = r.arc_elimination_time_s;
    out["greedy_time_s"] = r.greedy_time_s;
    out["frontier_lb_time_s"] = r.frontier_lb_time_s;
    out["postprocess_time_s"] = r.postprocess_time_s;
    out["n_labels_popped"] = r.n_labels_popped;
    out["n_arcs_considered"] = r.n_arcs_considered;
    out["n_pruned_arc"] = r.n_pruned_arc;
    out["n_pruned_elementary"] = r.n_pruned_elementary;
    out["n_pruned_capacity"] = r.n_pruned_capacity;
    out["n_dom_checks_forward"] = r.n_dom_checks_forward;
    out["n_dom_checks_reverse"] = r.n_dom_checks_reverse;
    out["max_bucket_size"] = r.max_bucket_size;
    out["n_final_labels"] = r.n_final_labels;
    out["n_pending_labels"] = r.n_pending_labels;
    out["dssr_iterations"] = r.dssr_iterations;
    out["n_customers_dropped"] = r.n_customers_dropped;
    out["ng_size"] = r.ng_size_used;
    out["bound_ng_size"] = r.bound_ng_size;
    out["bound_level"] = r.bound_level;
    out["n_escalations"] = r.n_escalations;
    out["root_lb"] = r.root_lb;
    out["relaxation_lb"] = r.relaxation_lb;
    out["lb_certified"] = (r.status == 0)
        || (r.status == 2 && std::isfinite(r.lb));
    if (r.status == 0) {
        out["status_reason"] = "optimal";
    } else if (r.timed_out) {
        out["status_reason"] = "time_limit";
    } else if (r.label_budget_exhausted) {
        out["status_reason"] = "label_budget";
    } else if (r.cutoff_bound_triggered) {
        out["status_reason"] = "cutoff_bound";
    } else if (r.status == 1) {
        out["status_reason"] = "infeasible";
    } else {
        out["status_reason"] = "interrupted";
    }
    return out;
}

} // namespace


PYBIND11_MODULE(espprc_cpp, m) {
    m.doc() = "ESPPRC forward-labeling for single-vehicle PCTSP (LSBC-strict).\n"
              "返回 top-K elementary depot->depot paths 按 path_cost 升序.";
    m.def("solve_espprc", &solve_espprc_py,
          py::arg("cost"),
          py::arg("pi_alpha"),
          py::arg("vol"),
          py::arg("n_active"),
          py::arg("depot_start"),
          py::arg("depot_end"),
          py::arg("capacity"),
          py::arg("cutoff") = INF,
          py::arg("top_k") = 5,
          py::arg("label_budget") = (uint64_t)0,
          py::arg("use_bidirectional") = true,
          R"pbdoc(
Solve single-vehicle ESPPRC by forward labeling + full elementary tracking.

Args
----
cost : (N, N) float64 row-major
    Arc costs c[i,j].
pi_alpha : (N,) float64
    Prize at each node (depots should be 0; inactive customers should NOT
    appear in this array -- only active customers + 2 depots).
vol : (N,) float64
    Resource usage per node (depots = 0).
n_active : int
    Number of active customers (must equal N-2 if you laid out
    [customers..., depot_start, depot_end]). Requires n_active <= 256.
depot_start, depot_end : int
    Indices of start/end depots.
capacity : float
    Vehicle capacity.
cutoff : float, default +inf
    Any label with cost >= cutoff is pruned (no impact on optimality below).
top_k : int, default 5
    Return up to top_k best paths.
label_budget : int, default 0 (=unlimited)
    Hard cap on total labels created. If reached, stops creating new ones
    and returns whatever has reached depot_end (status=2).

Returns
-------
dict with keys:
    paths : list of (cost, [node_index, ...]) sorted by cost ascending
    status : 0=optimal, 1=no feasible path, 2=label_budget triggered
    n_labels : total labels created (diagnostic)
    n_pruned_dom : labels rejected by dominance (diagnostic)
)pbdoc");

    m.def("solve_pctsp", &solve_pctsp_py,
          py::arg("cost"),
          py::arg("pi_alpha"),
          py::arg("vol"),
          py::arg("n_active"),
          py::arg("depot_start"),
          py::arg("depot_end"),
          py::arg("capacity"),
          py::arg("pi_y"),
          py::arg("vol_inactive"),
          py::arg("pi_alpha_inactive"),
          py::arg("cutoff") = INF,
          py::arg("top_k") = 5,
          py::arg("label_budget") = (uint64_t)0,
          py::arg("use_bidirectional") = true,
          py::arg("time_limit_s") = 0.0,
          py::arg("bi_probe_time_s") = 0.0,
          py::arg("ng_size") = 8,
          py::arg("bound_ng_size") = MAX_BOUND_NG,
          py::arg("bound_work_budget") = (uint64_t)200000000,
          R"pbdoc(
Solve single-vehicle PCTSP Lagrangian subproblem exactly.

Decomposes into ESPPRC (active customer routing) + 0-1 knapsack
(inactive customer assignment with remaining capacity) + vehicle
usage decision (yk=0 baseline).

The certifying core is an ng-route labeling with decremental state-space
relaxation (DSSR): each iteration solves an ng-route relaxation exactly
(its optimum is a certified lower bound), customers repeated by the relaxed
optimum become critical, and the loop stops when the relaxed optimum is
elementary.  Capacity-indexed q-route completion bounds drive best-first
search and interruption certificates.  ``ng_size`` is the neighbourhood
size of the relaxation (0 = plain q-route/DSSR, larger = tighter but more
labels per iteration).

Full objective (minimize):
    route_cost - knapsack_profit - pi_y * yk

Args
----
cost : (N, N) float64
    Arc costs for active customers + 2 depots.
pi_alpha : (N,) float64
    Lagrangian prizes for active customers (depots = 0).
vol : (N,) float64
    Volume per active customer (depots = 0).
n_active : int
    Number of active customers (N-2). Requires <= 256.
depot_start, depot_end : int
    Depot indices.
capacity : float
    Vehicle capacity (shared between routing and knapsack).
pi_y : float
    Lagrangian multiplier for vehicle usage.
vol_inactive : (n_inactive,) float64
    Volume per inactive customer.
pi_alpha_inactive : (n_inactive,) float64
    Lagrangian prizes for inactive customers.
cutoff : float, default +inf
    Prune solutions with full_obj >= cutoff.
top_k : int, default 5
    top_k<=1 preserves the former result fields. top_k>1 additionally exports
    at most min(top_k,64) saved elementary primal routes as candidate_routes;
    idle/knapsack-only states and duplicate active customer sets are excluded.
    Export scans at most 4096 candidates in the existing exact-objective order.
    Search/pruning and the single best solution/LB/status are unchanged. This
    is opportunistic primal output, not a complete or globally optimal top-K.
label_budget : int, default 0 (=unlimited)
    Hard cap on labels (summed over DSSR iterations).
use_bidirectional, bi_probe_time_s
    Accepted for API compatibility and ignored: the retired bidirectional
    solver was dominated by the ng/DSSR core on every measured instance.
time_limit_s : float, default 0 (=unlimited)
    Wall-clock limit; an interrupted call still returns a certified ``lb``.
ng_size : int, default 8
    Neighbourhood size of the ng-route relaxation (<= 8 keeps the archive
    slot index exact).
bound_ng_size : int, default 8
    Largest neighbourhood used by the completion-bound table.
bound_work_budget : int, default 2e8
    Cap on DP transitions per completion-bound table build.

Returns
-------
dict with keys:
    obj_val : float, best feasible PCTSP objective (UB on Q(π); == OPT if status==0)
    lb : float, valid lower bound on the optimal full_obj.
        - status==0 (exact)        ⇒ lb <= OPT <= obj_val; directed endpoints
          can differ by one binary64 ulp.
        - status==2 (time/budget)  ⇒ lb = max(last completed ng relaxation,
          current frontier key): 合法的 LB on Q(π), 可直接作为 outer cut 截距.
        - status==1 (infeasible)   ⇒ lb = +inf (vacuous).
    y : int, 0 or 1 (vehicle used)
    path : list of int, route node sequence (empty if y=0)
    alpha_inactive : list of int (0/1), inactive customer selection
    status : 0=optimal (also when interrupted with lb == obj_val),
             1=infeasible, 2=interrupted with a gap
    timed_out, label_budget_exhausted : bool, interruption cause
    dssr_iterations, n_customers_dropped, ng_size, bound_ng_size,
    bound_level, n_escalations, root_lb, relaxation_lb : core diagnostics
    n_labels, n_labels_popped, n_pruned_* , n_dom_checks_* : counters
    *_time_s : native timings (no Python overhead)
)pbdoc");
}
