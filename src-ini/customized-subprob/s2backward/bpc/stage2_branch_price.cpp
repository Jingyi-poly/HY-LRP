/*
 * stage2_branch_price.cpp
 *
 * Branch-Price-and-Cut (BPC) for the Stage-2 Lagrangian assignment subproblem.
 *
 * The master columns are vehicle-assignment patterns (v, S), where S is a
 * subset of active customers assigned to vehicle v and sum_{i in S} volume[i]
 * <= Qv[v]. The RMP keeps outsourcing variables o_i and theta variables. The
 * z variables are eliminated analytically from y_v <= z_v and -pi_z[v] z_v.
 *
 * Cutting planes generated internally:
 *   - Subset-Row Cuts (SR3): for triples {a,b,c} of active customers
 *   - Knapsack Cover Cuts: for vehicle-specific capacity covers
 *   - Capacity-Incompatible Clique Cuts: on outsourcing variables
 *
 * This file uses Gurobi only for the restricted master LP. Pricing and the
 * branch-and-price tree are implemented manually. Pricing is an exact 0-1
 * knapsack branch-and-bound under the current branching constraints.
 *
 * Compile sketch:
 *   g++ -O3 -std=c++17 -DSTAGE2_BP_DEMO stage2_branch_price.cpp \
 *       -I${GUROBI_HOME}/include -L${GUROBI_HOME}/lib \
 *       -lgurobi_c++ -lgurobi130 -o stage2_bp
 *
 * Adjust -lgurobi130 to your installed Gurobi version.
 */

#include <gurobi_c++.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace stage2bp {

static constexpr double INF = 1.0e100;
static constexpr double EPS = 1.0e-7;

namespace pricing_detail {

using UInt128 = unsigned __int128;

struct Binary64Positive {
    std::uint64_t significand = 0;
    int exponent = 0;
};

inline Binary64Positive decomposePositiveFinite(double value) {
    if (!(value > 0.0) || !std::isfinite(value)) {
        throw std::invalid_argument(
            "pricing density inputs must be positive and finite");
    }
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "unexpected binary64 size");
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint64_t fraction = bits & ((std::uint64_t{1} << 52) - 1);
    const int exponentBits = static_cast<int>((bits >> 52) & 0x7ffU);

    Binary64Positive result;
    if (exponentBits == 0) {
        result.significand = fraction;
        result.exponent = -1074;
    } else {
        result.significand = (std::uint64_t{1} << 52) | fraction;
        result.exponent = exponentBits - 1023 - 52;
    }
    while ((result.significand & 1U) == 0U) {
        result.significand >>= 1;
        ++result.exponent;
    }
    return result;
}

inline int bitWidth(UInt128 value) {
    const std::uint64_t high = static_cast<std::uint64_t>(value >> 64);
    if (high != 0) return 128 - __builtin_clzll(high);
    const std::uint64_t low = static_cast<std::uint64_t>(value);
    return low == 0 ? 0 : 64 - __builtin_clzll(low);
}

// Compare a*b with c*d exactly for positive finite binary64 inputs.  Each
// significand product needs at most 106 bits, so UInt128 retains every bit;
// exponents are compared separately without floating-point division.
inline int comparePositiveProducts(double a, double b, double c, double d) {
    const auto pa = decomposePositiveFinite(a);
    const auto pb = decomposePositiveFinite(b);
    const auto pc = decomposePositiveFinite(c);
    const auto pd = decomposePositiveFinite(d);
    UInt128 lhs = static_cast<UInt128>(pa.significand) * pb.significand;
    UInt128 rhs = static_cast<UInt128>(pc.significand) * pd.significand;
    const int lhsExponent = pa.exponent + pb.exponent;
    const int rhsExponent = pc.exponent + pd.exponent;
    const int lhsTop = bitWidth(lhs) + lhsExponent;
    const int rhsTop = bitWidth(rhs) + rhsExponent;
    if (lhsTop != rhsTop) return lhsTop < rhsTop ? -1 : 1;

    if (lhsExponent < rhsExponent) {
        rhs <<= (rhsExponent - lhsExponent);
    } else if (rhsExponent < lhsExponent) {
        lhs <<= (lhsExponent - rhsExponent);
    }
    if (lhs == rhs) return 0;
    return lhs < rhs ? -1 : 1;
}

// Negative additive costs are ordered by c/q ascending, equivalently by
// savings density (-c)/q descending.  Unlike the former 1e-12 ratio tie, this
// comparison is exact for the binary64 coefficients used by the model.
inline bool negativeCostDensityLess(double lhsCost, double lhsLoad,
                                    double rhsCost, double rhsLoad) {
    if (!(lhsLoad >= 0.0) || !std::isfinite(lhsLoad)
            || !(rhsLoad >= 0.0) || !std::isfinite(rhsLoad)
            || !std::isfinite(lhsCost) || !std::isfinite(rhsCost)) {
        throw std::invalid_argument("invalid pricing density input");
    }
    const bool lhsNegative = lhsCost < 0.0;
    const bool rhsNegative = rhsCost < 0.0;
    if (lhsNegative != rhsNegative) return lhsNegative;
    if (!lhsNegative) {
        if (lhsCost != rhsCost) return lhsCost < rhsCost;
        return lhsLoad < rhsLoad;
    }
    // A negative-cost zero-load item has infinite savings density and must be
    // consumed before every positive-load item.  Zero demand is legal input.
    if (lhsLoad == 0.0 || rhsLoad == 0.0) {
        if (lhsLoad == 0.0 && rhsLoad != 0.0) return true;
        if (rhsLoad == 0.0 && lhsLoad != 0.0) return false;
        if (lhsCost != rhsCost) return lhsCost < rhsCost;
        return false;
    }
    const int productComparison = comparePositiveProducts(
        -lhsCost, rhsLoad, -rhsCost, lhsLoad);
    if (productComparison != 0) return productComparison > 0;
    if (lhsCost != rhsCost) return lhsCost < rhsCost;
    return lhsLoad < rhsLoad;
}

inline double roundDown(double value) {
    return std::isfinite(value)
        ? std::nextafter(value, -std::numeric_limits<double>::infinity())
        : value;
}

inline double roundUp(double value) {
    return std::isfinite(value)
        ? std::nextafter(value, std::numeric_limits<double>::infinity())
        : value;
}

inline double addDown(double lhs, double rhs) {
    return roundDown(lhs + rhs);
}

inline double addUp(double lhs, double rhs) {
    return roundUp(lhs + rhs);
}

inline double multiplyDown(double lhs, double rhs) {
    return roundDown(lhs * rhs);
}

inline double subtractDown(double lhs, double rhs) {
    return roundDown(lhs - rhs);
}

inline double subtractUp(double lhs, double rhs) {
    return roundUp(lhs - rhs);
}

// Gurobi's Pi convention for a minimization model is Pi >= 0 on a >= row
// and Pi <= 0 on a <= row.  A wrong-sign finite value can always be projected
// to the valid endpoint zero; certification then has to re-price under that
// repaired dual.  `materialRepairs` is diagnostic only -- it never licenses
// an un-repaired dual for a certificate.
struct InequalityDualRepairStats {
    bool finite = true;
    int repairs = 0;
    int materialRepairs = 0;
};

inline double dualSignNoiseSlack(double value) {
    return 64.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, std::fabs(value));
}

inline void repairDualNonnegative(double& value,
                                  InequalityDualRepairStats& stats) {
    if (!std::isfinite(value)) {
        stats.finite = false;
        return;
    }
    if (value < 0.0) {
        if (-value > dualSignNoiseSlack(value)) ++stats.materialRepairs;
        value = 0.0;
        ++stats.repairs;
    }
}

inline void repairDualNonpositive(double& value,
                                  InequalityDualRepairStats& stats) {
    if (!std::isfinite(value)) {
        stats.finite = false;
        return;
    }
    if (value > 0.0) {
        if (value > dualSignNoiseSlack(value)) ++stats.materialRepairs;
        value = 0.0;
        ++stats.repairs;
    }
}

inline bool finiteVector(const std::vector<double>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); });
}

struct BinaryBigUInt {
    std::vector<std::uint64_t> limbs;  // little-endian base 2^64

    void addWord(std::size_t index, std::uint64_t word) {
        if (word == 0) return;
        if (limbs.size() <= index) limbs.resize(index + 1, 0);
        while (true) {
            const UInt128 sum = static_cast<UInt128>(limbs[index]) + word;
            limbs[index] = static_cast<std::uint64_t>(sum);
            word = static_cast<std::uint64_t>(sum >> 64);
            if (word == 0) break;
            ++index;
            if (limbs.size() <= index) limbs.push_back(0);
        }
    }

    void addShifted(std::uint64_t significand, unsigned shift) {
        const std::size_t index = shift / 64;
        const unsigned offset = shift % 64;
        const UInt128 shifted = static_cast<UInt128>(significand) << offset;
        addWord(index, static_cast<std::uint64_t>(shifted));
        addWord(index + 1, static_cast<std::uint64_t>(shifted >> 64));
    }

    void trim() {
        while (!limbs.empty() && limbs.back() == 0) limbs.pop_back();
    }
};

inline int compareBinaryBigUInt(BinaryBigUInt lhs, BinaryBigUInt rhs) {
    lhs.trim();
    rhs.trim();
    if (lhs.limbs.size() != rhs.limbs.size())
        return lhs.limbs.size() < rhs.limbs.size() ? -1 : 1;
    for (std::size_t index = lhs.limbs.size(); index > 0; --index) {
        const std::uint64_t a = lhs.limbs[index - 1];
        const std::uint64_t b = rhs.limbs[index - 1];
        if (a != b) return a < b ? -1 : 1;
    }
    return 0;
}

// Exact comparison of a nonnegative binary64 sum with a nonnegative binary64
// limit.  This is used only for numerically ambiguous capacity boundaries;
// ordinary cases are discharged by a cheap directed interval first.  Aligning
// every significand to the smallest binary exponent turns the comparison into
// integer arithmetic, so no epsilon can admit an overloaded route or reject a
// truly feasible boundary pattern.
inline int compareNonnegativeBinary64Sum(
        const std::vector<double>& values, double limit) {
    if (!(limit >= 0.0) || !std::isfinite(limit)) {
        throw std::invalid_argument("invalid exact-sum limit");
    }
    int minExponent = 0;
    bool havePositive = false;
    std::vector<Binary64Positive> parts;
    parts.reserve(values.size());
    for (double value : values) {
        if (!(value >= 0.0) || !std::isfinite(value)) {
            throw std::invalid_argument("invalid exact-sum term");
        }
        if (value == 0.0) continue;
        const auto part = decomposePositiveFinite(value);
        parts.push_back(part);
        if (!havePositive || part.exponent < minExponent)
            minExponent = part.exponent;
        havePositive = true;
    }
    Binary64Positive limitPart;
    if (limit > 0.0) {
        limitPart = decomposePositiveFinite(limit);
        if (!havePositive || limitPart.exponent < minExponent)
            minExponent = limitPart.exponent;
    }

    BinaryBigUInt sum;
    for (const auto& part : parts) {
        sum.addShifted(part.significand,
            static_cast<unsigned>(part.exponent - minExponent));
    }
    BinaryBigUInt rhs;
    if (limit > 0.0) {
        rhs.addShifted(limitPart.significand,
            static_cast<unsigned>(limitPart.exponent - minExponent));
    }
    return compareBinaryBigUInt(std::move(sum), std::move(rhs));
}

// Lower bound on the additional continuation cost of a density-sorted suffix.
// Branch restrictions and nonnegative nonlinear penalties are relaxed.  Every
// still-unrealized negative nonlinear contribution must be passed in via
// pendingNegativeBonus; including all of them is deliberately pessimistic and
// therefore safe even when they cannot be realized together.
template <class ItemContainer>
double fractionalContinuationLowerBound(
        const ItemContainer& items,
        int pos,
        double remainingCapacity,
        double pendingNegativeBonus,
        double capacityTolerance = EPS) {
    if (pos < 0 || pos > static_cast<int>(items.size())) {
        throw std::out_of_range("fractional continuation position out of range");
    }
    if (!std::isfinite(remainingCapacity)
            || !std::isfinite(pendingNegativeBonus)
            || !std::isfinite(capacityTolerance)
            || capacityTolerance < 0.0
            || pendingNegativeBonus > 0.0) {
        throw std::invalid_argument("invalid fractional continuation state");
    }

    double cap = roundUp(std::max(0.0, remainingCapacity) + capacityTolerance);
    double lowerBound = roundDown(pendingNegativeBonus);
    for (int index = pos; index < static_cast<int>(items.size()); ++index) {
        const double load = items[index].q;
        const double cost = items[index].c;
        if (!(load >= 0.0) || !std::isfinite(load) || !std::isfinite(cost)) {
            throw std::invalid_argument("invalid fractional continuation item");
        }
        if (cost >= 0.0) continue;
        if (load == 0.0) {
            lowerBound = addDown(lowerBound, cost);
            continue;
        }
        if (cap <= 0.0) continue;
        if (load <= cap) {
            lowerBound = addDown(lowerBound, cost);
            cap = roundUp(std::max(0.0, cap - load));
        } else {
            double fraction = roundUp(cap / load);
            fraction = std::min(1.0, std::max(0.0, fraction));
            const double fractionalCost = roundDown(cost * fraction);
            lowerBound = addDown(lowerBound, fractionalCost);
            break;
        }
    }
    return lowerBound;
}

}  // namespace pricing_detail

// One Stage-3 Benders/Lagrangian cut kept in the Stage-2 master.
// Row form in the RMP:
//   theta[succ] - sum_{v,S} a(cut,v,S) lambda[v,S] >= beta,
// where
//   a(cut,v,S) = piY[v] * 1{S != empty} + sum_{i in S} piAlpha[v][i].
struct Stage3Cut {
    int succ = 0;                         // successor/theta index, 0..numSucc-1
    double beta = 0.0;
    std::vector<double> piY;              // size m
    std::vector<std::vector<double>> piAlpha; // size m x nOriginal
};

struct Stage2Input {
    int n = 0;                            // number of original customers |J|
    int m = 0;                            // number of vehicles |V|
    int numSucc = 0;                      // number of theta variables

    std::vector<int> active;              // size n, 1 if active[j] = 1
    std::vector<double> volume;           // size n, node.volume[j]
    std::vector<double> cOut;             // size n, node.c_out[j]
    std::vector<double> Qv;               // size m, probData.Qv[v]
    std::vector<double> piZ;              // size m, pi_value["z[v,t]"]

    // Optional canonical vehicle-type id (size m).  When purchase-order
    // strengthening is enabled, equal ids define one ordered vehicle group
    // in the encounter order 0..m-1.  The RMP then represents the exact
    // convex hull of binary leading-prefix purchase vectors for each group.
    std::vector<int> vehicleType;

    std::vector<Stage3Cut> cuts;
};

struct Pattern {
    int v = -1;                           // vehicle index
    std::vector<int> items;               // active-index list, not original ids
    double load = 0.0;

    bool nonempty() const { return !items.empty(); }
};

struct Solution {
    bool feasible = false;
    double obj = INF;

    // Global lower bound on the Lagrangian subproblem optimum:
    //   - if optimality_proven == true, lb == obj;
    //   - otherwise, lb is the certified root LP relaxation objective, or
    //     -INF if exact pricing did not certify the root LP.
    // The lb field is what Python should use as V_k when building the
    // Lagrangian cut intercept (theta_h >= V_k + pi * z), so the cut stays
    // valid even when BP did not prove integer optimality.
    double lb = -INF;
    bool timed_out = false;                 // wall-clock / maxNodes / maxDepth / maxColgen hit
    int nodes_processed = 0;                // BB nodes explored

    // Certification flags.
    bool lb_certified = false;              // returned lb is mathematically valid
    bool ub_certified = false;              // incumbent passed full feasibility check
    bool optimality_proven = false;         // complete exact tree: lb == obj
    bool tree_complete = false;              // the complete B&P tree was fathomed
    bool intentional_root_stop = false;      // root-only mode stopped after certified root CG
    bool proof_relaxed = false;               // tolerance-based closure occurred; never exact
    int tolerance_bound_prunes = 0;
    int tolerance_integral_closures = 0;
    int pricing_certified_nodes = 0;
    int pricing_uncertified_nodes = 0;
    int used_inherited_lb_nodes = 0;

    // Instrumentation. Filled in by solve(). All times in seconds.
    double total_time_s = 0.0;              // wall-clock of solve()
    double rmp_build_time_s = 0.0;          // accumulated GRBModel build (addVar+addConstr+update)
    double rmp_solve_time_s = 0.0;          // accumulated GRBModel optimize()
    double pricing_time_s = 0.0;            // accumulated priceVehicle()
    double cut_separation_time_s = 0.0;
    double primal_heuristic_time_s = 0.0;
    int cg_iters_total = 0;                 // sum of CG iterations across all BB nodes
    int lp_solves = 0;                      // number of solveRmpLp() calls
    int columns_generated = 0;              // total columns added to allColumns_
    int exact_pricing_calls = 0;
    int pricing_deadline_interruptions = 0;
    int heuristic_pricing_successes = 0;
    int forward_cg_primal_calls = 0;
    int forward_cg_primal_improvements = 0;
    bool forward_reference_stop = false;
    bool backward_gap_stop = false;
    int forward_rounding_calls = 0;
    int forward_rounding_improvements = 0;
    int forward_partial_bound_calls = 0;
    int forward_partial_bound_certificates = 0;
    int forward_approximate_cg_returns = 0;

    // Root node LP statistics.
    double root_lp_before_cuts = INF;
    double root_lp_after_cuts = INF;
    int root_columns = 0;
    int root_cg_iters = 0;
    int root_pricing_passes_completed = 0;
    int root_cut_rounds_requested = 0;
    int root_cut_rounds_completed = 0;

    // Cut pool statistics.
    int num_sr3_active = 0;
    int num_cover_active = 0;
    int num_clique_active = 0;
    bool root_bound_only = false;
    bool sr3_cuts_enabled = true;
    bool cover_cuts_enabled = true;
    bool clique_cuts_enabled = true;
    bool purchase_order_enabled = false;

    // Why the solve stopped early, if it did. Empty when the tree was finished
    // cleanly. Set to a descriptive string when an internal exception was
    // swallowed as a "soft timeout" (this used to be hidden behind verbose).
    std::string abort_reason;

    std::vector<std::vector<int>> alpha;  // m x nOriginal, 0/1
    std::vector<int> y;                   // m, 0/1
    std::vector<int> z;                   // m, 0/1, reconstructed after z elimination
    std::vector<int> s;                   // nOriginal, 0/1 outsourcing
    std::vector<double> theta;            // numSucc
    std::vector<Pattern> chosenPatterns;
};

struct SolverParams {
    int maxDepth = 100000;
    int maxNodes = 1000000;
    int maxColgenIters = 1000;
    int pricingTopK = 5;
    double rcTol = 1.0e-7;
    double intTol = 1.0e-6;
    int strongBranchTopK = 3;
    int strongBranchMaxDepth = 5;
    int strongBranchMaxCG = 3;
    // ---- Cutting plane parameters ----
    int maxCuttingRounds = 10;       // max CG-cut rounds per BB node
    int maxSR3Cuts = 200;            // total SR3 cuts allowed (was 100)
    int maxCoverCuts = 100;          // total cover cuts allowed (was 50)
    int maxCliqueCuts = 10;          // total clique cuts allowed
    int maxSR3PerRound = 40;         // SR3 cuts added per separation round (was 20)
    int maxCoverPerRound = 10;       // cover cuts added per separation round
    double cutViolationTol = 1e-4;   // minimum violation to add a cut
    int maxCutDepth = 0;             // only separate cuts at root (was 3)
    bool useSR3Cuts = true;
    bool useCoverCuts = true;
    bool useCliqueCuts = true;
    // ---- Multi-level pricing ----
    bool useHeuristicPricing = true;
    // ---- Vehicle type clustering ----
    bool useVehicleClustering = true;
    // ---- Diving heuristic ----
    bool useDiving = true;
    int divingMaxDepth = 5;          // only dive at BB depth <= this
    // ---- Ryan-Foster branching ----
    bool useRyanFoster = true;
    // ---- Dual stabilization (Wentges) ----
    bool useDualStabilization = true;
    double smoothAlphaInit = 0.5;
    double smoothAlphaMin = 0.1;
    double smoothAlphaMax = 0.9;
    // ---- Column pool culling ----
    double cullRcThreshold = 10.0;
    // ---- LP method tuning ----
    int lpMethodThreshold = 500;     // switch to barrier above this many columns
    // ---- Adaptive pricingTopK ----
    int pricingTopKRoot = 30;
    int pricingTopKShallow = 20;
    int pricingTopKDeep = 5;
    int adaptiveTopKShallowDepth = 3;
    // ---- Restricted integer master heuristic ----
    bool useRestrictedMip = true;
    double restrictedMipTimeLimit = 0.5;
    int restrictedMipFreq = 5;
    // ---- Cut aging ----
    bool useCutAging = true;
    int cutAgingRounds = 10;
    double cutDualTol = 1e-8;
    // ---- Theta lower bound ----
    double thetaLowerBound = 0.0;
    // ---- Parallel pricing (OpenMP) ----
    int numThreads = 1;
    // Wall-clock soft cap.
    double timeLimitSec = -1.0;
    bool verbose = false;
    // ---- Forward (UB-focused) tuning ----
    // forwardGap > 0 enables MIPGap-style early termination: a node is pruned
    // once its (valid) LP lower bound is within forwardGap of the incumbent.
    // The incumbent is always validated by fullFeasibilityCheck so the returned
    // UB stays valid; only the optimality *proof* is relaxed (fine for SDDP
    // forward). The separately certified root LP remains a valid LB.
    double forwardGap = -1.0;
    // Backward query tolerance is a search stop, never an exact proof.
    // Both zero preserve the historical complete-search behavior.
    double backwardGapAbs = 0.0;
    double backwardGapRel = 0.0;
    // Caller-supplied certified relaxation value, used only as a forward UB
    // search target. It is never copied into the native pricing certificate.
    double forwardReferenceLb = -INF;
    double forwardPricingTimeSliceSec = 0.1;
#ifdef STAGE2BP_TESTING
    unsigned long long forwardPricingVisitLimit = 0;
#endif
    // preferRyanFoster makes branchAndPrice try the vehicle-agnostic
    // Ryan-Foster (customer-pair) branching BEFORE alpha[v] branching. This is
    // symmetry-free and avoids the vehicle-permutation blow-up on fleets with
    // many identical vehicles (HFVRP). alpha[v] branching is kept as fallback.
    bool preferRyanFoster = false;
    // forwardUbMode turns on the forward-only fast-UB behaviour (gated so
    // mode 0/2 are byte-for-byte unchanged):
    //   - depth>0 nodes use heuristic-only column generation (skip exact DFS),
    //     and their RMP obj is NEVER used as a lower bound;
    //   - a global "good enough" early stop fires once the incumbent is within
    //     forwardGap of the certified ROOT LP bound;
    //   - Ryan-Foster fallback to symmetric alpha[v] branching is suppressed
    //     once a feasible incumbent exists.
    bool forwardUbMode = false;
    // Backward-only fast path: solve and certify the root relaxation (including
    // every requested cut-and-price round), then return without entering the
    // B&P tree.  This is an intentional stop, not a timeout and not an
    // optimality proof unless the fully priced root solution itself is integer.
    bool rootBoundOnly = false;
    // Keep the free Stage-1 copy z in the canonical leading-prefix domain.
    // This only strengthens the relaxation; assignment ordering remains a
    // separate concern and is deliberately not inferred from capacity.
    bool usePurchaseOrder = false;
};

// Branching state. fixAlpha[v][a] uses active index a:
//   -1 free, 0 forbidden, 1 forced.
// fixY[v]: -1 free, 0 y_v=0, 1 y_v=1.
struct BranchState {
    std::vector<std::vector<int>> fixAlpha;
    std::vector<int> fixY;
    // Ryan-Foster pair branching on active customer indices:
    // - togetherPairs: alpha[v,a] == alpha[v,b] for all v (same vehicle or both outsourced)
    // - separatePairs: alpha[v,a] + alpha[v,b] <= 1 for all v (cannot share a vehicle)
    std::vector<std::pair<int,int>> togetherPairs;
    std::vector<std::pair<int,int>> separatePairs;
};

struct RmpResult {
    bool feasible = false;
    double obj = INF;

    // Never infer a mathematical bound from `obj`: it is the primal RMP
    // objective over the current subset of columns.  This separate field is
    // filled only after repairing/checking a raw RMP dual, complete pricing
    // over every vehicle's allowed pattern domain, and an explicit downward-
    // rounded dual-objective calculation.
    bool dualBoundCertified = false;
    double certifiedDualBound = -INF;

    std::vector<double> dualVehicle;
    std::vector<double> dualCustomer;     // active customer rows
    std::vector<double> dualCut;          // cuts rows, same order as input.cuts
    std::vector<double> dualSR3;          // SR3 cut rows
    std::vector<double> dualCover;        // cover cut rows
    std::vector<double> dualClique;       // clique cut rows
    std::vector<double> dualActivationLink;   // y_v - z_v <= 0
    std::vector<double> dualPurchaseConvexity;// one prefix per vehicle type

    std::vector<double> lambdaValue;      // one value per column
    std::vector<double> oValue;           // active customers
    std::vector<double> thetaValue;       // successors

    std::vector<std::vector<double>> alphaValue; // m x activeN
    std::vector<double> yValue;           // m
};

struct PricingColumn {
    double rc = INF;
    Pattern pattern;
};

struct PricingResult {
    std::vector<PricingColumn> columns;
    // Conservative lower envelope on the true minimum reduced cost over the
    // complete allowed pattern domain for this vehicle.  This is deliberately
    // allowed to be weaker than the best pattern found by DFS: convexity-dual
    // repair needs a lower bound, never a floating-point incumbent estimate.
    double minReducedCostLowerBound = -INF;
    bool exactDomainComplete = false;
    bool forwardTimeSliceExpired = false;
    bool globalDeadlineExpired = false;
    // Complete-domain lower envelope from directed frontier coverage, which
    // need not imply an exhaustive minimum-RC search or a complete kernel.
    bool reducedCostEnvelopeCertified = false;
    bool anyFound() const { return !columns.empty(); }
};

// The conservative DFS result before any vehicle-specific reduced-cost
// offset or negative-column filter is applied.  It is safe to share only
// under certificateSearchInputsBitwiseEqual().
struct PricingKernelEntry {
    double itemCost = INF;
    std::vector<int> items;
};

struct PricingKernelCapture {
    bool ready = false;
    std::vector<PricingKernelEntry> entries;
    // Complete conservative DFS envelope BEFORE patternAllowed column filtering.
    // A relaxed positive-cost fallback may be invalid as a column while still
    // being required to bound the omitted nonnegative-item search domain.
    double nonemptyItemCostLowerBound = INF;
};

struct DualCertificationAttempt {
    bool certified = false;
    bool pricingComplete = false;
    double bound = -INF;
    int newColumns = 0;
    pricing_detail::InequalityDualRepairStats signRepairs;
    std::string failureReason;
};

// Internal cut info structs
struct SR3CutInfo {
    int a, b, c;  // active customer indices (a < b < c)
    int age = 0;
    double lastDualAbs = 0.0;
    bool active = true;
};

struct CoverCutInfo {
    int vehicle;
    std::vector<int> cover;  // sorted active customer indices
    std::vector<int> liftCoeffs;  // size activeN_; lifted coefficient per item (0 if not in cut)
    int age = 0;
    double lastDualAbs = 0.0;
    bool active = true;
};

struct CliqueCutInfo {
    std::vector<int> clique;  // sorted active customer indices
};

class Stage2BranchPriceSolver {
public:
    explicit Stage2BranchPriceSolver(Stage2Input input, SolverParams params = SolverParams())
        : in_(std::move(input)), params_(params), env_(true) {
        validateInput();
        params_.numThreads = std::max(1, params_.numThreads);
        env_.set(GRB_IntParam_OutputFlag, 0);
        env_.set(GRB_IntParam_Threads, params_.numThreads);
        env_.start();
        preprocess();
    }

#ifdef STAGE2BP_TESTING
    // Narrow white-box hooks for adversarial certification regressions.  They
    // are not compiled into the Python extension or exposed to production.
    void testingResetPricingState() {
        t_start_ = std::chrono::steady_clock::now();
        allColumns_.clear();
        columnKeyToId_.clear();
        inColsStamp_.clear();
        colsGen_ = 0;
        sr3Cuts_.clear();
        coverCuts_.clear();
        cliqueCuts_.clear();
        sr3ByCust_.assign(activeN_, {});
        colsByCustomer_.assign(activeN_, {});
        colsByVehicle_.assign(in_.m, {});
        bumpInColsGen();
    }

    int testingSeedColumn(Pattern pattern, std::vector<int>& cols) {
        const int id = addColumnIfNew(pattern);
        if (!isInCols(id)) {
            cols.push_back(id);
            markInCols(id);
        }
        return id;
    }

    PricingResult testingPriceVehicle(
            int vehicle, const BranchState& state, const RmpResult& dual,
            int topK = 5, bool conservativeEnvelope = true) {
        precomputeCutAggregates(dual, conservativeEnvelope);
        return priceVehicle(
            vehicle, state, dual, topK, conservativeEnvelope);
    }

    DualCertificationAttempt testingCertifyRawDual(
            const BranchState& state, const RmpResult& dual,
            std::vector<int>& cols, int topK = 5) {
        return certifyRawDual(state, dual, cols, topK);
    }

    bool testingRepairRawDual(
            RmpResult& dual,
            pricing_detail::InequalityDualRepairStats& stats,
            std::vector<double>& thetaRcLower,
            std::string& reason) const {
        return repairRawDualForCertificate(
            dual, stats, thetaRcLower, reason);
    }

    bool testingQuickBranchFeasible(const BranchState& state) const {
        return quickBranchFeasible(state);
    }

    bool testingForwardRootDualBound(const BranchState& state,
                                    std::vector<int>& cols, RmpResult& rmp) {
        t_start_ = std::chrono::steady_clock::now();
        cg_iters_total_ = 20;
        last_forward_bound_cg_ = 0;
        last_forward_bound_time_ = -1.0;
        return forwardRootDualBound(state, cols, rmp, 0);
    }

    const Pattern& testingColumn(int id) const { return allColumns_.at(id); }
    bool testingProofRelaxed() const { return proof_relaxed_; }
    bool testingBackwardStop(bool rootCertified, double lower, bool feasible,
                             bool ubCertified, double upper) {
        root_lb_certified_ = rootCertified;
        global_lb_ = lower;
        best_.feasible = feasible;
        best_.ub_certified = ubCertified;
        best_.obj = upper;
        return backwardTargetSatisfied();
    }
    double testingGlobalLowerBound() const { return global_lb_; }
    bool testingLastCgComplete() const { return last_cg_certified_; }
#endif

    Solution solve() {
        best_ = Solution();
        nodesProcessed_ = 0;
        allColumns_.clear();
        columnKeyToId_.clear();
        timed_out_ = false;
        global_lb_ = -INF;
        root_lb_certified_ = false;
        rmp_build_time_s_ = 0.0;
        rmp_solve_time_s_ = 0.0;
        pricing_time_s_ = 0.0;
        cut_separation_time_s_ = 0.0;
        primal_heuristic_time_s_ = 0.0;
        cg_iters_total_ = 0;
        lp_solves_ = 0;
        exact_pricing_calls_ = 0;
        pricing_deadline_interruptions_ = 0;
        heuristic_pricing_successes_ = 0;
        pricing_certified_nodes_ = 0;
        pricing_uncertified_nodes_ = 0;
        used_inherited_lb_nodes_ = 0;
        used_gap_prune_ = false;
        proof_relaxed_ = false;
        tolerance_bound_prunes_ = 0;
        tolerance_integral_closures_ = 0;
        intentional_root_stop_ = false;
        root_integral_proof_ = false;
        root_lp_before_cuts_ = INF;
        root_lp_after_cuts_ = INF;
        root_columns_ = 0;
        root_cg_iters_ = 0;
        root_pricing_passes_completed_ = 0;
        root_cut_rounds_completed_ = 0;
        forward_cg_primal_calls_ = 0;
        forward_cg_primal_improvements_ = 0;
        last_forward_primal_cg_ = 0;
        last_forward_primal_time_ = 0.0;
        forward_reference_stop_ = false;
        backward_gap_stop_ = false;
        forward_rounding_calls_ = 0;
        forward_rounding_improvements_ = 0;
        forward_partial_bound_calls_ = 0;
        forward_partial_bound_certificates_ = 0;
        forward_approximate_cg_returns_ = 0;
        last_forward_bound_cg_ = 0;
        last_forward_bound_time_ = 0.0;
        t_start_ = std::chrono::steady_clock::now();
        rmpActiveMask_.clear();
        rmpActiveList_.clear();
        rmpWantScratch_.clear();
        colsGen_ = 0;
        inColsStamp_.clear();
        // Reset cutting plane state.
        sr3Cuts_.clear();
        coverCuts_.clear();
        cliqueCuts_.clear();
        sr3Rows_.clear();
        coverRows_.clear();
        cliqueRows_.clear();
        sr3ByCust_.assign(activeN_, {});
        colsByCustomer_.assign(activeN_, {});
        colsByVehicle_.assign(in_.m, {});
        // Build persistent RMP skeleton once. All future calls just toggle
        // column UBs and re-optimize.
        {
            const auto t0 = std::chrono::steady_clock::now();
            buildInitialRmp();
            rmp_build_time_s_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
        }
        // One solve budget includes the initial RMP build in every mode.

        BranchState root;
        root.fixAlpha.assign(in_.m, std::vector<int>(activeN_, -1));
        root.fixY.assign(in_.m, -1);
        root.togetherPairs.clear();
        root.separatePairs.clear();

        std::vector<int> rootColumns;
        ensureBasicColumns(root, rootColumns);

        // A cheap integer incumbent helps pruning.
        BranchState rootBs = root;
        Solution greedy = greedyIncumbent();
        if (greedy.feasible && fullFeasibilityCheck(greedy, rootBs)) {
            greedy.ub_certified = true;
            best_ = greedy;
        }
        if (params_.forwardUbMode && !params_.usePurchaseOrder)
            improveForwardRootPolicy(rootBs);

        // Branching loop. Internal "soft" termination (wall-clock, maxNodes,
        // maxDepth, maxColgen) just sets timed_out_ and bails out; only truly
        // unexpected errors propagate.
        try {
            if (!forwardReferenceSatisfied())
                branchAndPrice(root, rootColumns, 0);
        } catch (const GRBException& e) {
            // Gurobi errors do NOT derive from std::exception; catch them
            // explicitly so they are surfaced rather than aborting the process.
            timed_out_ = true;
            best_.abort_reason = "GRBException(" + std::to_string(e.getErrorCode())
                                 + "): " + e.getMessage();
            std::cerr << "  [stage2bp] branchAndPrice GRBException swallowed: "
                      << e.getMessage() << " (code " << e.getErrorCode() << ")\n";
        } catch (const std::exception& e) {
            // Treat any internal blow-up as a soft timeout so the caller still
            // gets a usable (incumbent, LB) pair instead of a hard exception.
            timed_out_ = true;
            best_.abort_reason = std::string("std::exception: ") + e.what();
            std::cerr << "  [stage2bp] branchAndPrice exception swallowed as soft timeout: "
                      << e.what() << "\n";
        } catch (...) {
            timed_out_ = true;
            best_.abort_reason = "unknown non-standard exception";
            std::cerr << "  [stage2bp] branchAndPrice unknown exception swallowed\n";
        }

        // A solve is exact only when the complete tree was fathomed.  An
        // intentional root-only return is therefore non-exact unless the
        // fully priced root RMP itself was integral and its reconstructed
        // incumbent matched the certified root bound.
        best_.intentional_root_stop = intentional_root_stop_;
        best_.proof_relaxed = proof_relaxed_;
        best_.tolerance_bound_prunes = tolerance_bound_prunes_;
        best_.tolerance_integral_closures = tolerance_integral_closures_;
        best_.tree_complete = (!timed_out_
                               && pricing_uncertified_nodes_ == 0
                               && !used_gap_prune_
                               && !proof_relaxed_
                               && (!intentional_root_stop_ || root_integral_proof_));
        best_.optimality_proven = (best_.tree_complete && best_.feasible);
        if (best_.optimality_proven) {
            best_.lb = best_.obj;
            best_.lb_certified = true;
        } else {
            best_.lb = global_lb_;
            // A child timeout, heuristic-only forward node, or gap prune does
            // not invalidate an already certified root relaxation.  Keep
            // validity separate from the stronger "tree solved" statement.
            best_.lb_certified = root_lb_certified_
                                 && global_lb_ > -INF + 1.0;
        }
        best_.timed_out = timed_out_;
        best_.nodes_processed = nodesProcessed_;
        best_.pricing_certified_nodes = pricing_certified_nodes_;
        best_.pricing_uncertified_nodes = pricing_uncertified_nodes_;
        best_.used_inherited_lb_nodes = used_inherited_lb_nodes_;
        best_.total_time_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start_).count();
        best_.rmp_build_time_s = rmp_build_time_s_;
        best_.rmp_solve_time_s = rmp_solve_time_s_;
        best_.pricing_time_s = pricing_time_s_;
        best_.cut_separation_time_s = cut_separation_time_s_;
        best_.primal_heuristic_time_s = primal_heuristic_time_s_;
        best_.cg_iters_total = cg_iters_total_;
        best_.lp_solves = lp_solves_;
        best_.columns_generated = (int)allColumns_.size();
        best_.exact_pricing_calls = exact_pricing_calls_;
        best_.pricing_deadline_interruptions = pricing_deadline_interruptions_;
        best_.heuristic_pricing_successes = heuristic_pricing_successes_;
        best_.forward_cg_primal_calls = forward_cg_primal_calls_;
        best_.forward_cg_primal_improvements = forward_cg_primal_improvements_;
        best_.forward_reference_stop = forward_reference_stop_;
        best_.backward_gap_stop = backward_gap_stop_;
        best_.forward_rounding_calls = forward_rounding_calls_;
        best_.forward_rounding_improvements = forward_rounding_improvements_;
        best_.forward_partial_bound_calls = forward_partial_bound_calls_;
        best_.forward_partial_bound_certificates = forward_partial_bound_certificates_;
        best_.forward_approximate_cg_returns = forward_approximate_cg_returns_;
        best_.root_lp_before_cuts = root_lp_before_cuts_;
        best_.root_lp_after_cuts = root_lp_after_cuts_;
        best_.root_columns = root_columns_;
        best_.root_cg_iters = root_cg_iters_;
        best_.root_pricing_passes_completed = root_pricing_passes_completed_;
        best_.root_cut_rounds_requested = params_.maxCuttingRounds;
        best_.root_cut_rounds_completed = root_cut_rounds_completed_;
        best_.num_sr3_active = (int)sr3Cuts_.size();
        best_.num_cover_active = (int)coverCuts_.size();
        best_.num_clique_active = (int)cliqueCuts_.size();
        best_.root_bound_only = params_.rootBoundOnly;
        best_.sr3_cuts_enabled = params_.useSR3Cuts;
        best_.cover_cuts_enabled = params_.useCoverCuts;
        best_.clique_cuts_enabled = params_.useCliqueCuts;
        best_.purchase_order_enabled = params_.usePurchaseOrder;
        return best_;
    }

private:
    Stage2Input in_;
    SolverParams params_;
    GRBEnv env_;

    std::vector<int> activeOrig_;          // active index -> original customer id
    std::vector<int> origToActive_;        // original customer id -> active index or -1
    int activeN_ = 0;
    bool integerVolumeSumsExact_ = false;  // all subset sums are exact binary64 integers

    double constantZ_ = 0.0;               // -sum_v max(piZ[v], 0)
    double inactiveOutCost_ = 0.0;         // sum inactive cOut[j], for exact match if inactive s_j=1 in Python MIP
    std::vector<double> deltaY_;           // max(-piZ[v], 0), coefficient of y_v after z elimination
    // Canonical purchase-prefix block. purchaseGroups_[g] is an ordered list
    // of vehicle positions. Prefix variable q means exactly ranks [0,q) are
    // purchased. This is the convex hull of z_r >= z_{r+1}, z binary.
    std::vector<std::vector<int>> purchaseGroups_;
    std::vector<int> purchaseGroupOfVehicle_;
    std::vector<int> purchaseRankOfVehicle_;

    std::vector<Pattern> allColumns_;
    std::unordered_map<std::string, int> columnKeyToId_;

    // O(1) "is id in the current cols vector?" check. Each `columnGenerationCore`
    // and `ensureBasicColumns` call bumps `colsGen_` and tags every id it
    // touches with the new generation; later membership tests just compare
    // `inColsStamp_[id] == colsGen_`. This replaces the O(|cols|) linear
    // `std::find` that was a real hot spot at large n (e.g. n=200 with
    // thousands of cached columns).
    int colsGen_ = 0;
    std::vector<int> inColsStamp_;

    void bumpInColsGen() {
        ++colsGen_;
        if (colsGen_ == 0) {
            std::fill(inColsStamp_.begin(), inColsStamp_.end(), 0);
            colsGen_ = 1;
        }
    }
    void markInCols(int id) {
        if (id < 0) return;
        if ((int)inColsStamp_.size() <= id) inColsStamp_.resize(id + 1, 0);
        inColsStamp_[id] = colsGen_;
    }
    bool isInCols(int id) const {
        if (id < 0 || id >= (int)inColsStamp_.size()) return false;
        return inColsStamp_[id] == colsGen_;
    }

    // ===== Incremental RMP state (persistent across BB tree) =====
    // Built once at the start of solve(); columns are added via GRBColumn
    // (column-wise insertion) and toggled active/inactive by setting UB = INF / 0.
    // colVars_ is parallel to allColumns_ (same indices).
    std::unique_ptr<GRBModel> rmpModel_;
    std::vector<GRBVar> colVars_;
    std::vector<GRBVar> oVars_;          // size activeN_
    std::vector<GRBVar> thetaVars_;      // size numSucc
    std::vector<GRBConstr> vehicleRows_; // size m
    std::vector<GRBConstr> customerRows_;// size activeN_
    std::vector<GRBConstr> cutRows_;     // size in_.cuts.size()
    std::vector<GRBConstr> activationLinkRows_; // y_v - z_v <= 0
    std::vector<GRBConstr> purchaseConvexityRows_;
    std::vector<std::vector<GRBVar>> purchasePrefixVars_;

    // ===== Cutting plane state (persistent across BB tree) =====
    std::vector<SR3CutInfo>    sr3Cuts_;
    std::vector<CoverCutInfo>  coverCuts_;
    std::vector<CliqueCutInfo> cliqueCuts_;
    std::vector<GRBConstr> sr3Rows_;
    std::vector<GRBConstr> coverRows_;
    std::vector<GRBConstr> cliqueRows_;
    // Per-customer index into sr3Cuts_. sr3ByCust_[a] = list of
    // (cutIndex, partner1, partner2) for efficient pricing DFS tracking.
    std::vector<std::vector<std::tuple<int,int,int>>> sr3ByCust_;

    // ===== Vehicle type clustering state =====
    std::vector<int> typeRep_;
    std::vector<std::vector<int>> typeMembers_;

    // ===== Dual stabilization (Wentges) state =====
    std::vector<double> stableDualVehicle_;
    std::vector<double> stableDualCustomer_;
    std::vector<double> stableDualCut_;
    std::vector<double> stableDualSR3_;
    std::vector<double> stableDualCover_;
    std::vector<double> stableDualActivationLink_;
    std::vector<double> stableDualPurchaseConvexity_;
    double stableObj_ = INF;
    bool stableInitialized_ = false;
    double smoothAlpha_ = 0.5;

    // ===== Cut aggregation state (precomputed per CG iteration) =====
    std::vector<double> aggActivation_;            // m
    std::vector<std::vector<double>> aggPiAlpha_;  // m x activeN_

    // ===== Sparse Benders cut index (precomputed once in preprocess) =====
    // cutSparseAlpha_[r][v] = list of (activeIndex, coefficient) for cut r, vehicle v
    std::vector<std::vector<std::vector<std::pair<int,double>>>> cutSparseAlpha_;

    // ===== Inverted index for column membership =====
    std::vector<std::vector<int>> colsByCustomer_;  // activeN_ → column ids containing customer
    std::vector<std::vector<int>> colsByVehicle_;   // m → column ids for vehicle v

    // ===== Bound tightening: compatibility matrix =====
    std::vector<std::vector<bool>> compatible_;  // m x activeN_

    // ===== LP method tuning state =====
    bool firstLpSolveInCG_ = true;

    Solution best_;
    int nodesProcessed_ = 0;

    // Soft-termination state.
    std::chrono::steady_clock::time_point t_start_;
    bool timed_out_ = false;
    // Global LB on the Stage-2 Lagrangian subproblem optimum. Set once the
    // root LP relaxation is solved; LP relaxations of child nodes only bound
    // their own subtree, so we conservatively keep the root LB as the global
    // tree LB. Remains -INF if even the root LP did not complete.
    double global_lb_ = -INF;
    bool root_lb_certified_ = false;

    // Instrumentation counters (accumulated during solve()).
    double rmp_build_time_s_ = 0.0;
    double rmp_solve_time_s_ = 0.0;
    double pricing_time_s_ = 0.0;
    double cut_separation_time_s_ = 0.0;
    double primal_heuristic_time_s_ = 0.0;
    int cg_iters_total_ = 0;
    int lp_solves_ = 0;
    int exact_pricing_calls_ = 0;
    int pricing_deadline_interruptions_ = 0;
    int heuristic_pricing_successes_ = 0;
    int pricing_certified_nodes_ = 0;
    int pricing_uncertified_nodes_ = 0;
    int used_inherited_lb_nodes_ = 0;
    bool used_gap_prune_ = false;
    bool proof_relaxed_ = false;
    int tolerance_bound_prunes_ = 0;
    int tolerance_integral_closures_ = 0;
    bool intentional_root_stop_ = false;
    bool root_integral_proof_ = false;
    // Forward UB mode: set true by the last columnGeneration call iff it
    // converged via EXACT pricing (a valid LP lower bound). When a node was
    // solved with heuristic-only pricing (forward fast path), this is false and
    // branchAndPrice must NOT treat rmp.obj as a lower bound.
    bool last_cg_certified_ = true;
    double root_lp_before_cuts_ = INF;
    double root_lp_after_cuts_ = INF;
    int root_columns_ = 0;
    int root_cg_iters_ = 0;
    int root_pricing_passes_completed_ = 0;
    int root_cut_rounds_completed_ = 0;
    int forward_cg_primal_calls_ = 0;
    int forward_cg_primal_improvements_ = 0;
    int last_forward_primal_cg_ = 0;
    double last_forward_primal_time_ = 0.0;
    bool forward_reference_stop_ = false;
    bool backward_gap_stop_ = false;
    int forward_rounding_calls_ = 0;
    int forward_rounding_improvements_ = 0;
    int forward_partial_bound_calls_ = 0;
    int forward_partial_bound_certificates_ = 0;
    int forward_approximate_cg_returns_ = 0;
    int last_forward_bound_cg_ = 0;
    double last_forward_bound_time_ = 0.0;

    bool backwardTargetSatisfied() {
        if (backward_gap_stop_) return true;
        if (params_.forwardUbMode
                || (params_.backwardGapAbs <= 0.0 && params_.backwardGapRel <= 0.0)
                || !root_lb_certified_ || !best_.feasible || !best_.ub_certified
                || !std::isfinite(global_lb_) || global_lb_ <= -INF + 1.0
                || !std::isfinite(best_.obj) || global_lb_ > best_.obj)
            return false;
        const double tolerance = std::max(params_.backwardGapAbs,
            params_.backwardGapRel * std::fabs(best_.obj));
        if (best_.obj - global_lb_ > tolerance) return false;
        // The raw incumbent only schedules this stop. Python independently
        // audits its complete policy and objective before accepting the query
        // interval. Keep the repaired global root LB, never promote this UB.
        backward_gap_stop_ = true;
        proof_relaxed_ = true;
        used_gap_prune_ = true;
        return true;
    }

    bool forwardReferenceSatisfied() {
        if (forward_reference_stop_) return true;
        const double reference = root_lb_certified_
            ? std::max(params_.forwardReferenceLb, global_lb_) : params_.forwardReferenceLb;
        if (!params_.forwardUbMode || params_.forwardGap <= 0.0
                || reference <= -INF + 1.0
                || !best_.feasible
                || reference > best_.obj) return false;
        if (best_.obj - reference
                > params_.forwardGap * std::max(1.0, std::fabs(best_.obj)))
            return false;
        forward_reference_stop_ = true;
        used_gap_prune_ = true;
        proof_relaxed_ = true;
        return true;
    }

    bool forwardRootDualBound(const BranchState& bs, std::vector<int>& cols,
                             RmpResult& rmp, int depth) {
        if (!params_.forwardUbMode || depth != 0 || !rmp.feasible
                || params_.forwardGap <= 0.0 || checkTimeout()) return false;
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start_).count();
        if (cg_iters_total_ - last_forward_bound_cg_ < 20
                || elapsed - last_forward_bound_time_ < 0.5) return false;
        const auto began = std::chrono::steady_clock::now();
        ++forward_partial_bound_calls_;
        // One minimum reduced cost per vehicle is sufficient for dual repair;
        // the ordinary CG oracle retains its configured multi-column TopK.
        const auto certificate = certifyRawDual(bs, rmp, cols, 1);
        pricing_time_s_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - began).count();
        last_forward_bound_cg_ = cg_iters_total_;
        last_forward_bound_time_ = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start_).count();
        // Newly generated columns have value zero in this still-feasible RMP
        // point. Keep lambda indexing aligned if the point exits to branching.
        rmp.lambdaValue.resize(cols.size(), 0.0);
        if (!certificate.certified) return false;
        ++forward_partial_bound_certificates_;
        global_lb_ = std::max(global_lb_, certificate.bound);
        root_lb_certified_ = true;
        root_lp_after_cuts_ = global_lb_;
        root_columns_ = static_cast<int>(cols.size());
        if (forwardReferenceSatisfied()) return true;
        // This forward-only LP-search tolerance is 1% of the unchanged
        // integer-policy target gap. It permits valid cut separation and
        // branching, but is never called exhaustive pricing or an exact CG.
        const double relaxationSlack = params_.forwardGap * 0.01
            * std::max(1.0, std::fabs(rmp.obj));
        if (rmp.obj - certificate.bound > relaxationSlack) return false;
        rmp.dualBoundCertified = true;
        rmp.certifiedDualBound = certificate.bound;
        last_cg_certified_ = false;
        proof_relaxed_ = true;
        ++forward_approximate_cg_returns_;
        return true;
    }

    // The legacy greedy seed ranks only outsourcing/volume and residual
    // capacity. Physical locations have different route-cut envelopes, so
    // improve that seed against the actual objective before spending the
    // entire forward budget certifying a column-generation relaxation.
    void improveForwardRootPolicy(const BranchState& bs) {
        if (!best_.feasible || checkTimeout() || forwardReferenceSatisfied()) return;
        const auto began = std::chrono::steady_clock::now();
        const double allowance = std::min(0.5, remainingTime() * 0.1);
        auto exhausted = [&]() {
            return forwardDeadlineExpired() ||
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - began).count() >= allowance;
        };
        while (!exhausted() && !forwardReferenceSatisfied()) {
            std::vector<int> owner(in_.n, -1), count(in_.m, 0);
            std::vector<double> load(in_.m, 0.0), rhs(in_.cuts.size(), 0.0);
            double outsourcing = 0.0;
            for (int j = 0; j < in_.n; ++j) {
                if (best_.s[j]) outsourcing += in_.cOut[j];
                for (int v = 0; v < in_.m; ++v) if (best_.alpha[v][j]) {
                    owner[j] = v;
                    ++count[v];
                    load[v] += in_.volume[j];
                }
            }
            for (int r = 0; r < (int)in_.cuts.size(); ++r) {
                const Stage3Cut& cut = in_.cuts[r];
                rhs[r] = cut.beta;
                for (int v = 0; v < in_.m; ++v) {
                    rhs[r] += cut.piY[v] * best_.y[v];
                    for (int j = 0; j < in_.n; ++j)
                        if (best_.alpha[v][j]) rhs[r] += cut.piAlpha[v][j];
                }
            }
            Solution improved = best_;
            auto consider = [&](int j, int to, int other = -1) {
                const int from = owner[j];
                if (from == to) return;
                auto counts = count;
                auto loads = load;
                double outCost = outsourcing;
                auto change = [&](int customer, int oldV, int newV) {
                    if (oldV >= 0) {
                        --counts[oldV]; loads[oldV] -= in_.volume[customer];
                    } else outCost -= in_.cOut[customer];
                    if (newV >= 0) {
                        ++counts[newV]; loads[newV] += in_.volume[customer];
                    } else outCost += in_.cOut[customer];
                };
                change(j, from, to);
                if (other >= 0) change(other, to, from);
                for (int v = 0; v < in_.m; ++v)
                    if (loads[v] > in_.Qv[v] + EPS) return;
                std::vector<double> theta(in_.numSucc,
                    std::max(0.0, params_.thetaLowerBound));
                for (int r = 0; r < (int)in_.cuts.size(); ++r) {
                    const Stage3Cut& cut = in_.cuts[r];
                    double value = rhs[r];
                    if (from >= 0) value -= cut.piAlpha[from][j];
                    if (to >= 0) value += cut.piAlpha[to][j];
                    if (other >= 0) {
                        if (to >= 0) value -= cut.piAlpha[to][other];
                        if (from >= 0) value += cut.piAlpha[from][other];
                    }
                    for (int v = 0; v < in_.m; ++v)
                        value += cut.piY[v] * ((counts[v] > 0) - best_.y[v]);
                    theta[cut.succ] = std::max(theta[cut.succ], value);
                }
                double score = outCost + constantZ_;
                for (double value : theta) score += value;
                for (int v = 0; v < in_.m; ++v)
                    if (counts[v]) score += deltaY_[v];
                // Floating incremental arithmetic ranks trials only. The
                // exact capacity and directed-up full-cut audit accepts them.
                if (score >= improved.obj - 1e-9) return;
                Solution candidate = best_;
                auto assign = [&](int customer, int newV) {
                    for (int v = 0; v < in_.m; ++v)
                        candidate.alpha[v][customer] = v == newV;
                    candidate.s[customer] = newV < 0;
                };
                assign(j, to);
                if (other >= 0) assign(other, from);
                for (int v = 0; v < in_.m; ++v) candidate.y[v] = counts[v] > 0;
                assignOptimalPurchaseVector(candidate);
                if (fullFeasibilityCheck(candidate, bs)
                        && candidate.obj < improved.obj - 1e-9) {
                    candidate.ub_certified = true;
                    improved = std::move(candidate);
                }
            };
            for (int a = 0; a < activeN_ && !exhausted(); ++a)
                for (int v = -1; v < in_.m; ++v)
                    consider(activeOrig_[a], v);
            if (improved.obj >= best_.obj - 1e-9) {
                for (int a = 0; a < activeN_ && !exhausted(); ++a)
                    for (int b = a + 1; b < activeN_; ++b) {
                        if ((b & 31) == 0 && exhausted()) break;
                        consider(activeOrig_[a], owner[activeOrig_[b]], activeOrig_[b]);
                    }
            }
            if (improved.obj >= best_.obj - 1e-9) break;
            best_ = std::move(improved);
        }
        primal_heuristic_time_s_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - began).count();
    }

    // Fractional assignment guidance complements whole-column diving: large
    // overlapping patterns can be useful for the LP but impossible to combine
    // into an integer restricted master. Build feasible assignments directly
    // from its alpha marginals, then audit the complete original cut envelope.
    void roundForwardRootAssignment(const BranchState& bs, const RmpResult& rmp) {
        if (!params_.forwardUbMode || !rmp.feasible
                || params_.usePurchaseOrder || checkTimeout()
                || rmp.alphaValue.size() != static_cast<std::size_t>(in_.m)) return;
        if (!bs.togetherPairs.empty() || !bs.separatePairs.empty()) return;
        for (int v = 0; v < in_.m; ++v) {
            if (bs.fixY[v] != -1) return;
            for (int a = 0; a < activeN_; ++a)
                if (bs.fixAlpha[v][a] != -1) return;
        }
        ++forward_rounding_calls_;
        const auto began = std::chrono::steady_clock::now();
        const double allowance = std::min(0.25, remainingTime() * 0.1);
        auto exhausted = [&]() {
            return forwardDeadlineExpired() ||
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - began).count() >= allowance;
        };
        std::vector<double> confidence(activeN_, 0.0), mass(activeN_, 0.0);
        for (int a = 0; a < activeN_; ++a)
            for (int v = 0; v < in_.m; ++v) {
                confidence[a] = std::max(confidence[a], rmp.alphaValue[v][a]);
                mass[a] += rmp.alphaValue[v][a];
            }
        bool improved = false;
        for (int variant = 0; variant < 4 && !exhausted(); ++variant) {
            std::vector<int> order(activeN_);
            std::iota(order.begin(), order.end(), 0);
            auto priority = [&](int a) {
                const int j = activeOrig_[a];
                if (variant == 0) return confidence[a];
                const double ratio = in_.cOut[j] / std::max(in_.volume[j], 1e-12);
                return ratio * (variant == 1 ? confidence[a] : variant == 2 ? mass[a] : 1.0);
            };
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                return priority(a) > priority(b);
            });
            Solution candidate;
            candidate.feasible = true;
            candidate.alpha.assign(in_.m, std::vector<int>(in_.n, 0));
            candidate.y.assign(in_.m, 0);
            candidate.z.assign(in_.m, 0);
            candidate.s.assign(in_.n, 1);
            std::vector<std::vector<int>> items(in_.m);
            std::vector<double> rhs(in_.cuts.size());
            std::vector<double> theta(in_.numSucc, std::max(0.0, params_.thetaLowerBound));
            for (int r = 0; r < (int)in_.cuts.size(); ++r) {
                rhs[r] = in_.cuts[r].beta;
                theta[in_.cuts[r].succ] = std::max(theta[in_.cuts[r].succ], rhs[r]);
            }
            for (int a : order) {
                if (exhausted()) break;
                const int j = activeOrig_[a];
                int chosen = -1;
                double bestScore = INF;
                std::vector<double> chosenTheta;
                const double oldTheta = std::accumulate(theta.begin(), theta.end(), 0.0);
                for (int v = 0; v < in_.m; ++v) {
                    if (!compatible_[v][a]
                            || !exactActiveLoadAtMostWithAdditional(items[v], {a}, in_.Qv[v])) continue;
                    std::vector<double> trialTheta(in_.numSucc,
                        std::max(0.0, params_.thetaLowerBound));
                    for (int r = 0; r < (int)in_.cuts.size(); ++r) {
                        const auto& cut = in_.cuts[r];
                        const double value = rhs[r] + cut.piAlpha[v][j]
                            + (candidate.y[v] ? 0.0 : cut.piY[v]);
                        trialTheta[cut.succ] = std::max(trialTheta[cut.succ], value);
                    }
                    const double delta = std::accumulate(trialTheta.begin(), trialTheta.end(), 0.0)
                        - oldTheta - in_.cOut[j] + (candidate.y[v] ? 0.0 : deltaY_[v]);
                    if (delta >= -1e-9) continue;
                    const double score = delta - (variant < 2
                        ? rmp.alphaValue[v][a] * std::max(0.0, in_.cOut[j]) : 0.0);
                    if (score < bestScore) {
                        bestScore = score; chosen = v; chosenTheta = std::move(trialTheta);
                    }
                }
                if (chosen < 0) continue;
                for (int r = 0; r < (int)in_.cuts.size(); ++r)
                    rhs[r] += in_.cuts[r].piAlpha[chosen][j]
                        + (candidate.y[chosen] ? 0.0 : in_.cuts[r].piY[chosen]);
                items[chosen].push_back(a);
                candidate.alpha[chosen][j] = 1;
                candidate.y[chosen] = 1;
                candidate.s[j] = 0;
                theta = std::move(chosenTheta);
            }
            assignOptimalPurchaseVector(candidate);
            if (fullFeasibilityCheck(candidate, bs) && candidate.obj < best_.obj - 1e-9) {
                candidate.ub_certified = true;
                best_ = std::move(candidate);
                ++forward_rounding_improvements_;
                improved = true;
            }
            if (forwardReferenceSatisfied()) break;
        }
        primal_heuristic_time_s_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - began).count();
        if (improved && !forwardReferenceSatisfied()) improveForwardRootPolicy(bs);
    }

    // CG need not close before its columns can improve a forward policy.
    // These integer masters are primal heuristics only: their objective and
    // solver bound never participate in native dual/pricing certification.
    void improveForwardDuringCg(const BranchState& bs,
                                const std::vector<int>& cols, bool force,
                                bool pricingYield = false,
                                const RmpResult* rmp = nullptr) {
        if (!params_.forwardUbMode || !params_.useRestrictedMip
                || checkTimeout() || forwardReferenceSatisfied()) return;
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start_).count();
        if (!force && ((!pricingYield && cg_iters_total_ - last_forward_primal_cg_ < 10)
                || elapsed - last_forward_primal_time_
                    < (forward_cg_primal_calls_ == 0 ? 0.25 : 1.0))) return;
        if (remainingTime() < 0.01) return;
        if (rmp) roundForwardRootAssignment(bs, *rmp);
        if (forwardReferenceSatisfied() || checkTimeout()) return;

        std::vector<int> primalCols = cols;
        // Keep the audited incumbent representable as a MIP start. This only
        // adds feasible patterns; it changes neither the pattern domain nor
        // the set of Benders constraints.
        Solution start = best_;
        if (start.feasible && fullFeasibilityCheck(start, bs)) {
            for (int v = 0; v < in_.m; ++v) {
                Pattern pattern;
                pattern.v = v;
                for (int a = 0; a < activeN_; ++a)
                    if (start.alpha[v][activeOrig_[a]]) pattern.items.push_back(a);
                pattern.load = patternLoad(pattern.items);
                const int id = addColumnIfNew(pattern);
                if (std::find(primalCols.begin(), primalCols.end(), id)
                        == primalCols.end()) primalCols.push_back(id);
            }
        }
        ++forward_cg_primal_calls_;
        Solution candidate = restrictedIntegerMaster(primalCols, bs);
        if (candidate.feasible && fullFeasibilityCheck(candidate, bs)) {
            candidate.ub_certified = true;
            if (!best_.feasible || candidate.obj < best_.obj) {
                best_ = std::move(candidate);
                ++forward_cg_primal_improvements_;
            }
        }
        last_forward_primal_cg_ = cg_iters_total_;
        last_forward_primal_time_ = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start_).count();
        forwardReferenceSatisfied();
    }

    double remainingTime() const {
        if (params_.timeLimitSec <= 0.0) return GRB_INFINITY;
        return std::max(0.0, params_.timeLimitSec -
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_start_).count());
    }

    // Const and local-state-only: pricing may run on OpenMP worker threads.
    bool globalDeadlineExpired() const {
        return remainingTime() <= 0.0;
    }

    bool forwardDeadlineExpired() const {
        return params_.forwardUbMode && globalDeadlineExpired();
    }

    // Lightweight soft-termination check. Returns true if BP should stop.
    // Pruning slack used to bound nodes against the incumbent.  The proof path
    // uses zero slack: LB >= UB is the only bound closure allowed to contribute
    // to `optimality_proven`.  A configured forwardGap is explicitly a relaxed
    // UB-search decision and permanently prevents an exact claim.
    double pruneSlack() const {
        if (params_.forwardGap > 0.0 && best_.feasible) {
            const double gapSlack = params_.forwardGap * std::max(1.0, std::fabs(best_.obj));
            if (gapSlack > 0.0) return gapSlack;
        }
        return 0.0;
    }

    bool shouldPruneByBound(double lowerBound) {
        if (!best_.feasible) return false;
        const double slack = pruneSlack();
        if (lowerBound < best_.obj - slack) return false;
        if (slack > 0.0) {
            used_gap_prune_ = true;
            proof_relaxed_ = true;
            ++tolerance_bound_prunes_;
        }
        return true;
    }

    bool checkTimeout() {
        if (timed_out_) return true;
        if (params_.timeLimitSec > 0.0) {
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_start_).count();
            if (elapsed > params_.timeLimitSec) {
                timed_out_ = true;
                return true;
            }
        }
        return false;
    }

private:
    void validateInput() const {
        if (in_.n <= 0 || in_.m <= 0 || in_.numSucc < 0)
            throw std::runtime_error("Invalid n, m, or numSucc.");
        if ((int)in_.active.size() != in_.n) throw std::runtime_error("active size mismatch.");
        if ((int)in_.volume.size() != in_.n) throw std::runtime_error("volume size mismatch.");
        if ((int)in_.cOut.size() != in_.n) throw std::runtime_error("cOut size mismatch.");
        if ((int)in_.Qv.size() != in_.m) throw std::runtime_error("Qv size mismatch.");
        if ((int)in_.piZ.size() != in_.m) throw std::runtime_error("piZ size mismatch.");
        if (params_.usePurchaseOrder
                && (int)in_.vehicleType.size() != in_.m) {
            throw std::runtime_error(
                "vehicleType size must equal m when purchase order is enabled.");
        }
        if (params_.usePurchaseOrder && !params_.rootBoundOnly) {
            throw std::runtime_error(
                "purchase-order strengthening is currently root-bound-only");
        }
        for (int j = 0; j < in_.n; ++j) {
            if (in_.active[j] != 0 && in_.active[j] != 1)
                throw std::runtime_error("active entries must be binary.");
            if (!std::isfinite(in_.volume[j]) || in_.volume[j] < 0.0)
                throw std::runtime_error("volume entries must be finite and nonnegative.");
            if (!std::isfinite(in_.cOut[j]))
                throw std::runtime_error("cOut entries must be finite.");
        }
        for (int v = 0; v < in_.m; ++v) {
            if (!std::isfinite(in_.Qv[v]) || in_.Qv[v] < 0.0)
                throw std::runtime_error("Qv entries must be finite and nonnegative.");
            if (!std::isfinite(in_.piZ[v]))
                throw std::runtime_error("piZ entries must be finite.");
        }
        for (const Stage3Cut& c : in_.cuts) {
            if (c.succ < 0 || c.succ >= in_.numSucc) throw std::runtime_error("cut succ out of range.");
            if (!std::isfinite(c.beta)) throw std::runtime_error("cut beta must be finite.");
            if ((int)c.piY.size() != in_.m) throw std::runtime_error("cut piY size mismatch.");
            if ((int)c.piAlpha.size() != in_.m) throw std::runtime_error("cut piAlpha vehicle size mismatch.");
            for (int v = 0; v < in_.m; ++v) {
                if (!std::isfinite(c.piY[v]))
                    throw std::runtime_error("cut piY entries must be finite.");
                if ((int)c.piAlpha[v].size() != in_.n) throw std::runtime_error("cut piAlpha customer size mismatch.");
                for (double coefficient : c.piAlpha[v]) {
                    if (!std::isfinite(coefficient))
                        throw std::runtime_error(
                            "cut piAlpha entries must be finite.");
                }
            }
        }
        if (!std::isfinite(params_.rcTol) || params_.rcTol < 0.0)
            throw std::runtime_error("rcTol must be finite and nonnegative.");
        if (!std::isfinite(params_.intTol)
                || params_.intTol < 0.0 || params_.intTol >= 0.5)
            throw std::runtime_error("intTol must be finite and in [0, 0.5).");
        if (!std::isfinite(params_.timeLimitSec))
            throw std::runtime_error("timeLimitSec must be finite.");
        if (!std::isfinite(params_.thetaLowerBound))
            throw std::runtime_error("thetaLowerBound must be finite.");
        if (!std::isfinite(params_.forwardGap))
            throw std::runtime_error("forwardGap must be finite.");
        if (!std::isfinite(params_.backwardGapAbs) || params_.backwardGapAbs < 0.0
                || !std::isfinite(params_.backwardGapRel) || params_.backwardGapRel < 0.0)
            throw std::runtime_error("backward gaps must be finite and nonnegative.");
        if (!std::isfinite(params_.cutViolationTol)
                || params_.cutViolationTol < 0.0)
            throw std::runtime_error(
                "cutViolationTol must be finite and nonnegative.");
        if (!std::isfinite(params_.restrictedMipTimeLimit)
                || params_.restrictedMipTimeLimit < 0.0)
            throw std::runtime_error(
                "restrictedMipTimeLimit must be finite and nonnegative.");
    }

    void preprocess() {
        // Fast exact-capacity path for Taillard and other integer-demand data.
        // Every nonnegative integer up to 2^53 and every subset sum whose total
        // stays below 2^53 is represented exactly by binary64 addition.
        pricing_detail::UInt128 integerVolumeTotal = 0;
        integerVolumeSumsExact_ = true;
        constexpr double MAX_EXACT_INTEGER = 9007199254740992.0;  // 2^53
        for (double volume : in_.volume) {
            if (!(volume >= 0.0) || !std::isfinite(volume)
                    || volume > MAX_EXACT_INTEGER
                    || std::floor(volume) != volume) {
                integerVolumeSumsExact_ = false;
                break;
            }
            integerVolumeTotal += static_cast<std::uint64_t>(volume);
            if (integerVolumeTotal
                    > static_cast<pricing_detail::UInt128>(
                        std::uint64_t{1} << 53)) {
                integerVolumeSumsExact_ = false;
                break;
            }
        }

        origToActive_.assign(in_.n, -1);
        activeOrig_.clear();
        inactiveOutCost_ = 0.0;
        for (int j = 0; j < in_.n; ++j) {
            if (in_.active[j]) {
                origToActive_[j] = (int)activeOrig_.size();
                activeOrig_.push_back(j);
            } else {
                inactiveOutCost_ += in_.cOut[j];
            }
        }
        activeN_ = (int)activeOrig_.size();

        // Bound tightening: build compatible_[v][a] and remove customers
        // that don't fit in any vehicle.
        compatible_.assign(in_.m, std::vector<bool>(activeN_, false));
        for (int v = 0; v < in_.m; ++v)
            for (int a = 0; a < activeN_; ++a)
                compatible_[v][a] =
                    (in_.volume[activeOrig_[a]] <= in_.Qv[v]);

        std::vector<int> newActiveOrig;
        std::vector<int> newOrigToActive(in_.n, -1);
        for (int a = 0; a < activeN_; ++a) {
            bool fits = false;
            for (int v = 0; v < in_.m; ++v) {
                if (compatible_[v][a]) { fits = true; break; }
            }
            if (fits) {
                newOrigToActive[activeOrig_[a]] = (int)newActiveOrig.size();
                newActiveOrig.push_back(activeOrig_[a]);
            } else {
                inactiveOutCost_ += in_.cOut[activeOrig_[a]];
            }
        }
        if ((int)newActiveOrig.size() < activeN_) {
            activeOrig_ = std::move(newActiveOrig);
            origToActive_ = std::move(newOrigToActive);
            activeN_ = (int)activeOrig_.size();
            compatible_.assign(in_.m, std::vector<bool>(activeN_, false));
            for (int v = 0; v < in_.m; ++v)
                for (int a = 0; a < activeN_; ++a)
                    compatible_[v][a] =
                        (in_.volume[activeOrig_[a]] <= in_.Qv[v]);
        }

        purchaseGroups_.clear();
        purchaseGroupOfVehicle_.assign(in_.m, -1);
        purchaseRankOfVehicle_.assign(in_.m, -1);
        if (params_.usePurchaseOrder) {
            std::unordered_map<int, int> groupByType;
            for (int v = 0; v < in_.m; ++v) {
                const int type = in_.vehicleType[v];
                auto [it, inserted] = groupByType.emplace(
                    type, static_cast<int>(purchaseGroups_.size()));
                if (inserted) purchaseGroups_.push_back({});
                const int group = it->second;
                purchaseGroupOfVehicle_[v] = group;
                purchaseRankOfVehicle_[v] =
                    static_cast<int>(purchaseGroups_[group].size());
                purchaseGroups_[group].push_back(v);
            }
        }

        constantZ_ = 0.0;
        deltaY_.assign(in_.m, 0.0);
        if (!params_.usePurchaseOrder) {
            for (int v = 0; v < in_.m; ++v) {
                constantZ_ += -std::max(in_.piZ[v], 0.0);
                deltaY_[v] = std::max(-in_.piZ[v], 0.0);
            }
        }

        // Build sparse Benders cut index for fast aggregation
        cutSparseAlpha_.resize(in_.cuts.size());
        for (int r = 0; r < (int)in_.cuts.size(); ++r) {
            cutSparseAlpha_[r].resize(in_.m);
            for (int v = 0; v < in_.m; ++v) {
                for (int a = 0; a < activeN_; ++a) {
                    double coeff = in_.cuts[r].piAlpha[v][activeOrig_[a]];
                    if (std::fabs(coeff) > 1e-15) {
                        cutSparseAlpha_[r][v].emplace_back(a, coeff);
                    }
                }
            }
        }
    }

    std::string patternKey(const Pattern& p) const {
        std::ostringstream oss;
        oss << p.v << ":";
        for (int a : p.items) oss << a << ",";
        return oss.str();
    }

    double purchasePrefixCost(int group, int prefixLength) const {
        double cost = 0.0;
        const auto& vehicles = purchaseGroups_.at(group);
        for (int rank = 0; rank < prefixLength; ++rank)
            cost -= in_.piZ[vehicles[rank]];
        return cost;
    }

    double activationReducedCostTerm(
            int v, const RmpResult& rmp,
            bool conservativeEnvelope = false) const {
        double term = conservativeEnvelope
            ? pricing_detail::addDown(deltaY_[v], aggActivation_[v])
            : deltaY_[v] + aggActivation_[v];
        if (params_.usePurchaseOrder) {
            const double link = rmp.dualActivationLink[v];
            term = conservativeEnvelope
                ? pricing_detail::addDown(term, -link)
                : term - link;
        }
        return term;
    }

    void assignOptimalPurchaseVector(Solution& sol) const {
        sol.z.assign(in_.m, 0);
        if (!params_.usePurchaseOrder) {
            for (int v = 0; v < in_.m; ++v) {
                sol.z[v] = sol.y[v] == 1 || in_.piZ[v] > 0.0 ? 1 : 0;
            }
            return;
        }
        for (int group = 0;
                group < static_cast<int>(purchaseGroups_.size()); ++group) {
            const auto& vehicles = purchaseGroups_[group];
            int required = 0;
            for (int rank = 0; rank < static_cast<int>(vehicles.size()); ++rank) {
                if (sol.y[vehicles[rank]]) required = rank + 1;
            }
            int bestPrefix = required;
            double bestCost = purchasePrefixCost(group, required);
            for (int prefix = required + 1;
                    prefix <= static_cast<int>(vehicles.size()); ++prefix) {
                const double cost = purchasePrefixCost(group, prefix);
                if (cost < bestCost) {
                    bestCost = cost;
                    bestPrefix = prefix;
                }
            }
            for (int rank = 0; rank < bestPrefix; ++rank)
                sol.z[vehicles[rank]] = 1;
        }
    }

    bool containsActive(const Pattern& p, int a) const {
        return std::binary_search(p.items.begin(), p.items.end(), a);
    }

    static std::pair<int,int> normPair(int a, int b) {
        if (a > b) std::swap(a, b);
        return {a, b};
    }

    double patternLoad(const std::vector<int>& items) const {
        double load = 0.0;
        for (int a : items) load += in_.volume[activeOrig_[a]];
        return load;
    }

    bool exactActiveLoadAtMost(
            const std::vector<int>& items, double capacity) const {
        double lower = 0.0;
        double upper = 0.0;
        for (int a : items) {
            const double volume = in_.volume[activeOrig_[a]];
            lower = pricing_detail::addDown(lower, volume);
            upper = pricing_detail::addUp(upper, volume);
        }
        if (upper <= capacity) return true;
        if (lower > capacity) return false;
        std::vector<double> ambiguousTerms;
        ambiguousTerms.reserve(items.size());
        for (int a : items)
            ambiguousTerms.push_back(in_.volume[activeOrig_[a]]);
        return pricing_detail::compareNonnegativeBinary64Sum(
            ambiguousTerms, capacity) <= 0;
    }

    bool exactActiveLoadAtMostWithAdditional(
            const std::vector<int>& base,
            const std::vector<int>& additional,
            double capacity) const {
        double lower = 0.0;
        double upper = 0.0;
        auto accumulate = [&](const std::vector<int>& values) {
            for (int a : values) {
                const double volume = in_.volume[activeOrig_[a]];
                lower = pricing_detail::addDown(lower, volume);
                upper = pricing_detail::addUp(upper, volume);
            }
        };
        accumulate(base);
        accumulate(additional);
        if (upper <= capacity) return true;
        if (lower > capacity) return false;
        std::vector<double> ambiguousTerms;
        ambiguousTerms.reserve(base.size() + additional.size());
        for (int a : base)
            ambiguousTerms.push_back(in_.volume[activeOrig_[a]]);
        for (int a : additional)
            ambiguousTerms.push_back(in_.volume[activeOrig_[a]]);
        return pricing_detail::compareNonnegativeBinary64Sum(
            ambiguousTerms, capacity) <= 0;
    }

    bool patternAllowed(const Pattern& p, const BranchState& bs) const {
        if (p.v < 0 || p.v >= in_.m) return false;
        if (integerVolumeSumsExact_) {
            if (p.load > in_.Qv[p.v]) return false;
        } else if (!exactActiveLoadAtMost(p.items, in_.Qv[p.v])) {
            return false;
        }
        if (bs.fixY[p.v] == 0 && p.nonempty()) return false;
        if (bs.fixY[p.v] == 1 && !p.nonempty()) return false;

        for (int a = 0; a < activeN_; ++a) {
            const bool has = containsActive(p, a);
            if (bs.fixAlpha[p.v][a] == 0 && has) return false;
            if (bs.fixAlpha[p.v][a] == 1 && !has) return false;
        }

        // Ryan-Foster branching constraints.
        for (const auto& pr : bs.togetherPairs) {
            const bool ha = containsActive(p, pr.first);
            const bool hb = containsActive(p, pr.second);
            if (ha != hb) return false; // must appear together on each vehicle pattern
        }
        for (const auto& pr : bs.separatePairs) {
            const bool ha = containsActive(p, pr.first);
            const bool hb = containsActive(p, pr.second);
            if (ha && hb) return false; // forbidden together on the same vehicle pattern
        }
        return true;
    }

    int addColumnIfNew(const Pattern& p) {
        Pattern q = p;
        std::sort(q.items.begin(), q.items.end());
        q.items.erase(std::unique(q.items.begin(), q.items.end()), q.items.end());
        q.load = patternLoad(q.items);

        std::string key = patternKey(q);
        auto it = columnKeyToId_.find(key);
        if (it != columnKeyToId_.end()) return it->second;

        const int id = (int)allColumns_.size();
        allColumns_.push_back(q);
        columnKeyToId_[key] = id;
        if ((int)inColsStamp_.size() <= id) inColsStamp_.resize(id + 1, 0);

        colsByVehicle_[q.v].push_back(id);
        for (int a : q.items) colsByCustomer_[a].push_back(id);

        // Mirror the new column into the persistent RMP model via column-wise
        // insertion. The new variable starts with UB = 0 (disabled). solveRmpLp
        // will toggle UB = INF for columns in the active set `cols`.
        if (rmpModel_) {
            GRBColumn col;
            col.addTerm(1.0, vehicleRows_[q.v]);
            for (int a : q.items) col.addTerm(1.0, customerRows_[a]);
            for (int r = 0; r < (int)in_.cuts.size(); ++r) {
                const double a = cutCoeff(in_.cuts[r], q);
                if (a != 0.0) col.addTerm(-a, cutRows_[r]);
            }
            if (params_.usePurchaseOrder && q.nonempty()) {
                col.addTerm(1.0, activationLinkRows_[q.v]);
            }
            // SR3 cut coefficients: 1 if |pattern ∩ {a,b,c}| >= 2
            for (int ci = 0; ci < (int)sr3Cuts_.size(); ++ci) {
                const auto& sc = sr3Cuts_[ci];
                int cnt = 0;
                if (containsActive(q, sc.a)) ++cnt;
                if (containsActive(q, sc.b)) ++cnt;
                if (containsActive(q, sc.c)) ++cnt;
                if (cnt >= 2) col.addTerm(1.0, sr3Rows_[ci]);
            }
            // Cover cut coefficients (with lifting)
            for (int ci = 0; ci < (int)coverCuts_.size(); ++ci) {
                const auto& cc = coverCuts_[ci];
                if (cc.vehicle != q.v) continue;
                int coeff = 0;
                if (cc.liftCoeffs.empty()) {
                    for (int ca : cc.cover) {
                        if (containsActive(q, ca)) ++coeff;
                    }
                } else {
                    for (int ia : q.items) {
                        if (ia < (int)cc.liftCoeffs.size() && cc.liftCoeffs[ia] > 0)
                            coeff += cc.liftCoeffs[ia];
                    }
                }
                if (coeff > 0) col.addTerm((double)coeff, coverRows_[ci]);
            }
            // Clique cuts only involve o variables, no lambda coefficient.
            const double obj = deltaY_[q.v] * (q.nonempty() ? 1.0 : 0.0);
            GRBVar var = rmpModel_->addVar(
                0.0, 0.0, obj, GRB_CONTINUOUS,
                col, std::string("lam_" + std::to_string(id)));
            colVars_.push_back(var);
        }
        return id;
    }

    // Build the persistent RMP skeleton (o, theta, vehicle/customer/cut rows).
    // No columns are added here -- they are added on-demand via addColumnIfNew.
    void buildInitialRmp() {
        rmpModel_ = std::make_unique<GRBModel>(env_);
        rmpModel_->set(GRB_IntParam_OutputFlag, 0);
        rmpModel_->set(GRB_IntParam_Method, 1); // dual simplex
        rmpModel_->set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);

        oVars_.clear(); oVars_.reserve(activeN_);
        for (int a = 0; a < activeN_; ++a) {
            const int j = activeOrig_[a];
            oVars_.push_back(rmpModel_->addVar(
                0.0, GRB_INFINITY, in_.cOut[j],
                GRB_CONTINUOUS, "out_" + std::to_string(j)));
        }
        thetaVars_.clear(); thetaVars_.reserve(in_.numSucc);
        for (int h = 0; h < in_.numSucc; ++h) {
            thetaVars_.push_back(rmpModel_->addVar(
                params_.thetaLowerBound, GRB_INFINITY, 1.0,
                GRB_CONTINUOUS, "theta_" + std::to_string(h)));
        }
        rmpModel_->update();

        vehicleRows_.clear(); vehicleRows_.reserve(in_.m);
        for (int v = 0; v < in_.m; ++v) {
            vehicleRows_.push_back(rmpModel_->addConstr(
                GRBLinExpr() == 1.0, "choose_vehicle_" + std::to_string(v)));
        }
        customerRows_.clear(); customerRows_.reserve(activeN_);
        for (int a = 0; a < activeN_; ++a) {
            customerRows_.push_back(rmpModel_->addConstr(
                GRBLinExpr(oVars_[a]) == 1.0,
                "fulfill_" + std::to_string(activeOrig_[a])));
        }
        cutRows_.clear(); cutRows_.reserve(in_.cuts.size());
        for (int r = 0; r < (int)in_.cuts.size(); ++r) {
            const Stage3Cut& cut = in_.cuts[r];
            cutRows_.push_back(rmpModel_->addConstr(
                GRBLinExpr(thetaVars_[cut.succ]) >= cut.beta,
                "s3cut_" + std::to_string(r)));
        }
        rmpModel_->update();

        activationLinkRows_.clear();
        purchaseConvexityRows_.clear();
        purchasePrefixVars_.clear();
        if (params_.usePurchaseOrder) {
            activationLinkRows_.reserve(in_.m);
            for (int v = 0; v < in_.m; ++v) {
                activationLinkRows_.push_back(rmpModel_->addConstr(
                    GRBLinExpr() <= 0.0,
                    "purchase_link_" + std::to_string(v)));
            }
            purchaseConvexityRows_.reserve(purchaseGroups_.size());
            for (int group = 0;
                    group < static_cast<int>(purchaseGroups_.size()); ++group) {
                purchaseConvexityRows_.push_back(rmpModel_->addConstr(
                    GRBLinExpr() == 1.0,
                    "purchase_prefix_" + std::to_string(group)));
            }
            rmpModel_->update();

            purchasePrefixVars_.resize(purchaseGroups_.size());
            for (int group = 0;
                    group < static_cast<int>(purchaseGroups_.size()); ++group) {
                const auto& vehicles = purchaseGroups_[group];
                auto& variables = purchasePrefixVars_[group];
                variables.reserve(vehicles.size() + 1);
                for (int prefix = 0;
                        prefix <= static_cast<int>(vehicles.size()); ++prefix) {
                    GRBColumn column;
                    column.addTerm(1.0, purchaseConvexityRows_[group]);
                    for (int rank = 0; rank < prefix; ++rank) {
                        column.addTerm(-1.0,
                                      activationLinkRows_[vehicles[rank]]);
                    }
                    variables.push_back(rmpModel_->addVar(
                        0.0, GRB_INFINITY,
                        purchasePrefixCost(group, prefix),
                        GRB_CONTINUOUS, column,
                        "purchase_prefix_var_" + std::to_string(group)
                            + "_" + std::to_string(prefix)));
                }
            }
            rmpModel_->update();
        }
        colVars_.clear();
    }

    // Reset all column UBs to 0 (deactivate). Called between solve() runs.
    void resetRmpModel() {
        rmpModel_.reset();
        colVars_.clear();
        oVars_.clear();
        thetaVars_.clear();
        vehicleRows_.clear();
        customerRows_.clear();
        cutRows_.clear();
        activationLinkRows_.clear();
        purchaseConvexityRows_.clear();
        purchasePrefixVars_.clear();
    }

    void pushColumnIfAllowed(const Pattern& p, const BranchState& bs, std::vector<int>& cols) {
        Pattern q = p;
        std::sort(q.items.begin(), q.items.end());
        q.items.erase(std::unique(q.items.begin(), q.items.end()), q.items.end());
        q.load = patternLoad(q.items);
        if (!patternAllowed(q, bs)) return;
        int id = addColumnIfNew(q);
        if (!isInCols(id)) {
            cols.push_back(id);
            markInCols(id);
        }
    }

    void ensureBasicColumns(const BranchState& bs, std::vector<int>& cols) {
        // Retain only columns allowed by this node.
        std::vector<int> filtered;
        filtered.reserve(cols.size());
        for (int id : cols) {
            if (id >= 0 && id < (int)allColumns_.size() && patternAllowed(allColumns_[id], bs)) {
                filtered.push_back(id);
            }
        }
        cols.swap(filtered);

        // Refresh the O(1) "in cols" membership stamp for this call.
        bumpInColsGen();
        for (int id : cols) markInCols(id);

        for (int v = 0; v < in_.m; ++v) {
            // Forced set for vehicle v.
            std::vector<int> forced;
            for (int a = 0; a < activeN_; ++a) {
                if (bs.fixAlpha[v][a] == 1) forced.push_back(a);
            }

            // Empty pattern if allowed.
            Pattern empty;
            empty.v = v;
            pushColumnIfAllowed(empty, bs, cols);

            // Minimal forced pattern if there are forced customers.
            if (!forced.empty()) {
                Pattern p;
                p.v = v;
                p.items = forced;
                pushColumnIfAllowed(p, bs, cols);
            }

            // Singleton or forced+one columns.
            for (int a = 0; a < activeN_; ++a) {
                if (bs.fixAlpha[v][a] == 0) continue;
                if (!compatible_[v][a]) continue;
                Pattern p;
                p.v = v;
                p.items = forced;
                if (std::find(p.items.begin(), p.items.end(), a) == p.items.end()) p.items.push_back(a);
                pushColumnIfAllowed(p, bs, cols);
            }

            // Ryan-Foster together support columns: forced + (a,b).
            // Without these seed columns, a together-branch child can look
            // infeasible before pricing has a chance to generate pair columns.
            for (const auto& pr : bs.togetherPairs) {
                const int a = pr.first, b = pr.second;
                if (bs.fixAlpha[v][a] == 0 || bs.fixAlpha[v][b] == 0) continue;
                Pattern p;
                p.v = v;
                p.items = forced;
                if (std::find(p.items.begin(), p.items.end(), a) == p.items.end()) p.items.push_back(a);
                if (std::find(p.items.begin(), p.items.end(), b) == p.items.end()) p.items.push_back(b);
                pushColumnIfAllowed(p, bs, cols);
            }
        }
    }

    double cutCoeff(const Stage3Cut& cut, const Pattern& p) const {
        double a = 0.0;
        if (p.nonempty()) a += cut.piY[p.v];
        for (int activeIdx : p.items) {
            const int j = activeOrig_[activeIdx];
            a += cut.piAlpha[p.v][j];
        }
        return a;
    }

    // ----------------------------------------------------------------------
    // Incremental RMP solve.
    //
    // The model is built once (in solve()) and persists across all BB nodes.
    // Each call here just toggles which columns are active (UB = INF) vs
    // disabled (UB = 0) for this BB node, then re-optimizes warm-starting
    // from the previous dual basis.
    //
    // We assume colVars_.size() == allColumns_.size() at all times (every
    // pattern that ever enters allColumns_ also gets a corresponding GRBVar
    // via addColumnIfNew). Columns NOT in `cols` get UB = 0; columns IN
    // `cols` get UB = INF.
    //
    // For efficiency we cache the previous active set in lastActiveCols_
    // and only toggle the symmetric difference, instead of touching all
    // colVars_ on every call.
    // ----------------------------------------------------------------------
    // Activation mask + the ids that are currently UB=INF, kept in sync with
    // the persistent RMP model. Maintaining the active list lets us compute
    // the symmetric difference vs the next `cols` in O(|active| + |cols|)
    // instead of scanning all of colVars_ on every LP solve, which matters
    // at n=200 / m=32 where the pool can hold many thousands of columns but
    // only a few hundred are active in any given BB node.
    std::vector<bool> rmpActiveMask_;     // size = allColumns_.size(); true = currently UB=INF
    std::vector<int>  rmpActiveList_;     // ids with rmpActiveMask_[id]=true (unordered)
    std::vector<char> rmpWantScratch_;    // scratch buffer reused across calls
    void setRmpActiveSet(const std::vector<int>& cols) {
        const int N = (int)colVars_.size();
        if ((int)rmpActiveMask_.size() < N) rmpActiveMask_.resize(N, false);
        if ((int)rmpWantScratch_.size() < N) rmpWantScratch_.resize(N, 0);

        // Mark which ids should be active after this call.
        for (int id : cols) {
            if (id >= 0 && id < N) rmpWantScratch_[id] = 1;
        }

        // 1) Deactivate ids that are currently active but not wanted.
        //    Compact `rmpActiveList_` in place by keeping only still-wanted
        //    entries; the rest get UB=0.
        int write = 0;
        for (int id : rmpActiveList_) {
            if (id < 0 || id >= N) continue;
            if (rmpWantScratch_[id]) {
                rmpActiveList_[write++] = id;
            } else {
                colVars_[id].set(GRB_DoubleAttr_UB, 0.0);
                rmpActiveMask_[id] = false;
            }
        }
        rmpActiveList_.resize(write);

        // 2) Activate ids that are wanted but currently inactive.
        for (int id : cols) {
            if (id < 0 || id >= N) continue;
            if (!rmpActiveMask_[id]) {
                colVars_[id].set(GRB_DoubleAttr_UB, GRB_INFINITY);
                rmpActiveMask_[id] = true;
                rmpActiveList_.push_back(id);
            }
        }

        // 3) Clear the scratch marks we set (only those ids).
        for (int id : cols) {
            if (id >= 0 && id < N) rmpWantScratch_[id] = 0;
        }
    }

    RmpResult solveRmpLp(const std::vector<int>& cols) {
        RmpResult res;
        ++lp_solves_;
        const auto t_build0 = std::chrono::steady_clock::now();
        try {
            if (!rmpModel_) {
                throw std::runtime_error("solveRmpLp called before buildInitialRmp");
            }
            // Ensure every id in cols has a GRBVar; addColumnIfNew is the
            // single entry point that keeps colVars_ in sync with allColumns_.
            // We don't add columns here -- the caller (CG / ensureBasicColumns)
            // is expected to have routed all new patterns through
            // addColumnIfNew already.

            setRmpActiveSet(cols);

            // LP method tuning: barrier for large first-solves, dual simplex for warm-starts
            int activeCols = (int)rmpActiveList_.size();
            if (firstLpSolveInCG_ && activeCols > params_.lpMethodThreshold) {
                rmpModel_->set(GRB_IntParam_Method, 2);  // barrier
            } else {
                rmpModel_->set(GRB_IntParam_Method, 1);  // dual simplex (warm-start)
            }

            const auto t_solve0 = std::chrono::steady_clock::now();
            rmp_build_time_s_ += std::chrono::duration<double>(t_solve0 - t_build0).count();
            if (checkTimeout()) return res;
            rmpModel_->set(GRB_DoubleParam_TimeLimit, remainingTime());
            rmpModel_->optimize();
            rmp_solve_time_s_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_solve0).count();

            const int status = rmpModel_->get(GRB_IntAttr_Status);
            if (status == GRB_TIME_LIMIT) {
                timed_out_ = true;
                return res;
            }
            if (status == GRB_INFEASIBLE || status == GRB_INF_OR_UNBD || status == GRB_UNBOUNDED) {
                res.feasible = false;
                return res;
            }
            if (status != GRB_OPTIMAL) {
                throw std::runtime_error("RMP LP did not solve to optimality; status = " + std::to_string(status));
            }

            res.feasible = true;
            res.obj = rmpModel_->get(GRB_DoubleAttr_ObjVal) + constantZ_ + inactiveOutCost_;

            res.dualVehicle.assign(in_.m, 0.0);
            for (int v = 0; v < in_.m; ++v) res.dualVehicle[v] = vehicleRows_[v].get(GRB_DoubleAttr_Pi);

            res.dualCustomer.assign(activeN_, 0.0);
            for (int a = 0; a < activeN_; ++a) res.dualCustomer[a] = customerRows_[a].get(GRB_DoubleAttr_Pi);

            res.dualCut.assign(in_.cuts.size(), 0.0);
            for (int r = 0; r < (int)in_.cuts.size(); ++r) res.dualCut[r] = cutRows_[r].get(GRB_DoubleAttr_Pi);

            // Extract internal cut duals.
            res.dualSR3.assign(sr3Cuts_.size(), 0.0);
            for (int ci = 0; ci < (int)sr3Cuts_.size(); ++ci)
                res.dualSR3[ci] = sr3Rows_[ci].get(GRB_DoubleAttr_Pi);
            res.dualCover.assign(coverCuts_.size(), 0.0);
            for (int ci = 0; ci < (int)coverCuts_.size(); ++ci)
                res.dualCover[ci] = coverRows_[ci].get(GRB_DoubleAttr_Pi);
            res.dualClique.assign(cliqueCuts_.size(), 0.0);
            for (int ci = 0; ci < (int)cliqueCuts_.size(); ++ci)
                res.dualClique[ci] = cliqueRows_[ci].get(GRB_DoubleAttr_Pi);

            res.dualActivationLink.assign(
                params_.usePurchaseOrder ? in_.m : 0, 0.0);
            for (int v = 0;
                    v < static_cast<int>(res.dualActivationLink.size()); ++v) {
                res.dualActivationLink[v] =
                    activationLinkRows_[v].get(GRB_DoubleAttr_Pi);
            }
            res.dualPurchaseConvexity.assign(
                params_.usePurchaseOrder ? purchaseGroups_.size() : 0, 0.0);
            for (int group = 0;
                    group < static_cast<int>(
                        res.dualPurchaseConvexity.size()); ++group) {
                res.dualPurchaseConvexity[group] =
                    purchaseConvexityRows_[group].get(GRB_DoubleAttr_Pi);
            }

            const int C = (int)cols.size();
            res.lambdaValue.assign(C, 0.0);
            for (int k = 0; k < C; ++k) {
                res.lambdaValue[k] = colVars_[cols[k]].get(GRB_DoubleAttr_X);
            }

            res.oValue.assign(activeN_, 0.0);
            for (int a = 0; a < activeN_; ++a) res.oValue[a] = oVars_[a].get(GRB_DoubleAttr_X);

            res.thetaValue.assign(in_.numSucc, 0.0);
            for (int h = 0; h < in_.numSucc; ++h) res.thetaValue[h] = thetaVars_[h].get(GRB_DoubleAttr_X);

            res.alphaValue.assign(in_.m, std::vector<double>(activeN_, 0.0));
            res.yValue.assign(in_.m, 0.0);
            for (int k = 0; k < C; ++k) {
                const double val = res.lambdaValue[k];
                if (std::fabs(val) <= 1e-12) continue;
                const Pattern& p = allColumns_[cols[k]];
                if (p.nonempty()) res.yValue[p.v] += val;
                for (int a : p.items) res.alphaValue[p.v][a] += val;
            }
        } catch (const GRBException& e) {
            throw std::runtime_error(std::string("Gurobi error in solveRmpLp: ") + e.getMessage());
        }
        return res;
    }

    PricingResult priceVehicle(int v, const BranchState& bs, const RmpResult& rmp,
                              int topK = -1,
                              bool conservativeEnvelope = false,
                              PricingKernelCapture* kernelCapture = nullptr) const {
        if (kernelCapture) *kernelCapture = PricingKernelCapture{};
        PricingResult result;
        if (globalDeadlineExpired()) {
            result.globalDeadlineExpired = true;
            return result;
        }
        const auto pricingStarted = std::chrono::steady_clock::now();
        if (topK <= 0) topK = params_.pricingTopK;

        // Empty is an ordinary pattern and must participate in certification.
        // In particular, fixY=0 leaves exactly this column, and a duplicate
        // empty column with a small negative reduced cost must still lower the
        // certified dual objective even though it cannot be added again.
        Pattern emptyPattern;
        emptyPattern.v = v;
        const bool emptyAllowed = patternAllowed(emptyPattern, bs);
        const double emptyRcLower = pricing_detail::roundDown(-rmp.dualVehicle[v]);
        if (bs.fixY[v] == 0) {
            if (emptyAllowed) {
                result.minReducedCostLowerBound = emptyRcLower;
                result.exactDomainComplete = true;
            }
            return result;
        }

        double activationCoeff = activationReducedCostTerm(
            v, rmp, conservativeEnvelope);

        std::vector<double> itemCoeff(activeN_, 0.0);
        for (int a = 0; a < activeN_; ++a) {
            itemCoeff[a] = conservativeEnvelope
                ? pricing_detail::addDown(-rmp.dualCustomer[a], aggPiAlpha_[v][a])
                : -rmp.dualCustomer[a] + aggPiAlpha_[v][a];
        }

        // Cover cut dual adjustments (additive per-item, with lifting coeffs).
        for (int ci = 0; ci < (int)coverCuts_.size(); ++ci) {
            const auto& cc = coverCuts_[ci];
            if (cc.vehicle != v) continue;
            const double sigma = rmp.dualCover[ci];
            if (cc.liftCoeffs.empty()) {
                for (int ca : cc.cover) {
                    itemCoeff[ca] = conservativeEnvelope
                        ? pricing_detail::addDown(itemCoeff[ca], -sigma)
                        : itemCoeff[ca] - sigma;
                }
            } else {
                for (int a = 0; a < activeN_; ++a) {
                    if (cc.liftCoeffs[a] > 0) {
                        const double term = conservativeEnvelope
                            ? pricing_detail::multiplyDown(
                                -sigma, static_cast<double>(cc.liftCoeffs[a]))
                            : -sigma * cc.liftCoeffs[a];
                        itemCoeff[a] = conservativeEnvelope
                            ? pricing_detail::addDown(itemCoeff[a], term)
                            : itemCoeff[a] + term;
                    }
                }
            }
        }

        std::vector<int> forced;
        std::vector<int> cand;
        double forcedLoad = 0.0;
        double forcedBoundLoad = 0.0;
        double forcedLoadLower = 0.0;
        double forcedLoadUpper = 0.0;
        double forcedCost = 0.0;
        auto extendBoundLoad = [&](double current, double volume) {
            if (!conservativeEnvelope || integerVolumeSumsExact_)
                return current + volume;
            return std::max(
                0.0, pricing_detail::addDown(current, volume));
        };
        auto extendCapacityInterval = [&](double& lower, double& upper,
                                          double volume) {
            if (!conservativeEnvelope || integerVolumeSumsExact_) {
                lower += volume;
                upper = lower;
                return;
            }
            lower = std::max(
                0.0, pricing_detail::addDown(lower, volume));
            upper = pricing_detail::addUp(upper, volume);
        };
        for (int a = 0; a < activeN_; ++a) {
            if (bs.fixAlpha[v][a] == 0 || !compatible_[v][a]) continue;
            if (bs.fixAlpha[v][a] == 1) {
                forced.push_back(a);
                const double volume = in_.volume[activeOrig_[a]];
                forcedLoad += volume;
                forcedBoundLoad = extendBoundLoad(forcedBoundLoad, volume);
                extendCapacityInterval(
                    forcedLoadLower, forcedLoadUpper, volume);
                forcedCost = conservativeEnvelope
                    ? pricing_detail::addDown(forcedCost, itemCoeff[a])
                    : forcedCost + itemCoeff[a];
            } else {
                cand.push_back(a);
            }
        }

        // Ryan-Foster enforcement: remove from cand any item that has a
        // separate-pair with a forced item.
        if (!bs.separatePairs.empty()) {
            std::vector<bool> excluded(activeN_, false);
            for (int fa : forced) {
                for (const auto& pr : bs.separatePairs) {
                    if (pr.first == fa) excluded[pr.second] = true;
                    else if (pr.second == fa) excluded[pr.first] = true;
                }
            }
            std::vector<int> filteredCand;
            filteredCand.reserve(cand.size());
            for (int a : cand) {
                if (!excluded[a]) filteredCand.push_back(a);
            }
            cand.swap(filteredCand);
        }

        // Ryan-Foster together-pairs: merge items into super-items.
        // Items that must appear together form groups; treat each group atomically.
        // Build union-find over together-pairs restricted to cand set.
        std::vector<int> parent(activeN_, -1);
        for (int a : cand) parent[a] = a;
        for (int fa : forced) parent[fa] = fa;
        auto ufFind = [&](int x) {
            while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
            return x;
        };
        auto ufUnion = [&](int x, int y) {
            x = ufFind(x); y = ufFind(y);
            if (x != y) parent[x] = y;
        };
        for (const auto& pr : bs.togetherPairs) {
            if (parent[pr.first] >= 0 && parent[pr.second] >= 0)
                ufUnion(pr.first, pr.second);
        }

        // Check if any cand item is together-paired with a forced item → becomes forced
        for (const auto& pr : bs.togetherPairs) {
            bool firstForced = (bs.fixAlpha[v][pr.first] == 1);
            bool secondForced = (bs.fixAlpha[v][pr.second] == 1);
            if (firstForced && !secondForced && parent[pr.second] >= 0) {
                auto it = std::find(cand.begin(), cand.end(), pr.second);
                if (it != cand.end()) {
                    cand.erase(it);
                    forced.push_back(pr.second);
                    const double volume = in_.volume[activeOrig_[pr.second]];
                    forcedLoad += volume;
                    forcedBoundLoad = extendBoundLoad(
                        forcedBoundLoad, volume);
                    extendCapacityInterval(
                        forcedLoadLower, forcedLoadUpper, volume);
                    forcedCost = conservativeEnvelope
                        ? pricing_detail::addDown(forcedCost, itemCoeff[pr.second])
                        : forcedCost + itemCoeff[pr.second];
                }
            } else if (secondForced && !firstForced && parent[pr.first] >= 0) {
                auto it = std::find(cand.begin(), cand.end(), pr.first);
                if (it != cand.end()) {
                    cand.erase(it);
                    forced.push_back(pr.first);
                    const double volume = in_.volume[activeOrig_[pr.first]];
                    forcedLoad += volume;
                    forcedBoundLoad = extendBoundLoad(
                        forcedBoundLoad, volume);
                    extendCapacityInterval(
                        forcedLoadLower, forcedLoadUpper, volume);
                    forcedCost = conservativeEnvelope
                        ? pricing_detail::addDown(forcedCost, itemCoeff[pr.first])
                        : forcedCost + itemCoeff[pr.first];
                }
            }
        }

        // Build super-items from together-pair groups among remaining cand
        struct SuperItem {
            std::vector<int> members;
            double totalVol = 0.0;
            double boundVol = 0.0;
            double loadLower = 0.0;
            double loadUpper = 0.0;
            double totalCost = 0.0;
        };
        std::unordered_map<int, SuperItem> superMap;
        for (int a : cand) {
            int root = ufFind(a);
            superMap[root].members.push_back(a);
            const double volume = in_.volume[activeOrig_[a]];
            superMap[root].totalVol += volume;
            superMap[root].boundVol = extendBoundLoad(
                superMap[root].boundVol, volume);
            extendCapacityInterval(
                superMap[root].loadLower,
                superMap[root].loadUpper,
                volume);
            superMap[root].totalCost = conservativeEnvelope
                ? pricing_detail::addDown(superMap[root].totalCost, itemCoeff[a])
                : superMap[root].totalCost + itemCoeff[a];
        }

        // SR3 penalty tracking: initialize counts from forced items.
        std::vector<int> sr3Count(sr3Cuts_.size(), 0);
        double forcedSR3Penalty = 0.0;
        for (int fa : forced) {
            for (const auto& [ci, p1, p2] : sr3ByCust_[fa]) {
                ++sr3Count[ci];
                if (sr3Count[ci] == 2) {
                    forcedSR3Penalty = conservativeEnvelope
                        ? pricing_detail::addDown(
                            forcedSR3Penalty, -rmp.dualSR3[ci])
                        : forcedSR3Penalty + (-rmp.dualSR3[ci]);
                }
            }
        }
        forcedCost = conservativeEnvelope
            ? pricing_detail::addDown(forcedCost, forcedSR3Penalty)
            : forcedCost + forcedSR3Penalty;

        // A <= SR3 row should have a nonpositive dual in a minimization LP,
        // hence its pricing contribution -dual is normally nonnegative.  Do
        // not assume perfect numerical sign feasibility, though: a positive
        // dual creates a negative pair/triple bonus.  Keep every unrealized
        // such bonus in the continuation LB and retain nonnegative-cost groups
        // that could help realize one; otherwise exact pricing could miss a
        // negative reduced-cost column.
        double pendingNegativeSR3Bonus = 0.0;
        std::vector<bool> negativeSR3Relevant(activeN_, false);
        for (int ci = 0; ci < static_cast<int>(sr3Cuts_.size()); ++ci) {
            const double penalty = -rmp.dualSR3[ci];
            if (penalty >= 0.0 || sr3Count[ci] >= 2) continue;
            pendingNegativeSR3Bonus = pricing_detail::addDown(
                pendingNegativeSR3Bonus, penalty);
            const auto& cut = sr3Cuts_[ci];
            negativeSR3Relevant[cut.a] = true;
            negativeSR3Relevant[cut.b] = true;
            negativeSR3Relevant[cut.c] = true;
        }

        bool forcedCapacityFeasible = false;
        if (integerVolumeSumsExact_) {
            forcedCapacityFeasible = forcedLoad <= in_.Qv[v];
        } else if (forcedLoadUpper <= in_.Qv[v]) {
            forcedCapacityFeasible = true;
        } else if (forcedLoadLower > in_.Qv[v]) {
            forcedCapacityFeasible = false;
        } else {
            forcedCapacityFeasible = exactActiveLoadAtMost(
                forced, in_.Qv[v]);
        }
        if (!forcedCapacityFeasible) return result;
        if (bs.fixY[v] == 1 && forced.empty() && cand.empty()) return result;

        // Build the DFS item array from super-items.
        // Each super-item is an atomic unit (together-pair group).
        // Negative-cost items drive the knapsack search. If none can be selected,
        // a positive-cost singleton/group can still have negative reduced cost
        // once the fixed activation term is included, so keep the best fallback.
        // In the conservative certificate oracle `q` is a downward envelope
        // on the super-item's exact load.  Together with an upward envelope on
        // remaining capacity this makes the fractional-knapsack continuation
        // a true relaxation even when ordinary load summation rounds upward.
        struct ItemView {
            std::vector<int> members;
            double q;
            double loadLower;
            double loadUpper;
            double c;
        };
        std::vector<ItemView> items;
        items.reserve(superMap.size());
        ItemView bestNonNegativeFallback;
        bool hasNonNegativeFallback = false;

        // Build separate-pair lookup for DFS pruning
        std::vector<std::vector<int>> separatePartners(activeN_);
        for (const auto& pr : bs.separatePairs) {
            separatePartners[pr.first].push_back(pr.second);
            separatePartners[pr.second].push_back(pr.first);
        }

        for (auto& [root, si] : superMap) {
            bool canRealizeNegativeSR3 = false;
            for (int member : si.members) {
                if (negativeSR3Relevant[member]) {
                    canRealizeNegativeSR3 = true;
                    break;
                }
            }
            const bool nonnegativeForSearch = conservativeEnvelope
                ? si.totalCost >= 0.0 : si.totalCost >= -1e-12;
            if (nonnegativeForSearch && !canRealizeNegativeSR3) {
                if (forced.empty()
                        && (integerVolumeSumsExact_
                            ? si.totalVol <= in_.Qv[v]
                            : exactActiveLoadAtMost(si.members, in_.Qv[v]))
                        && (!hasNonNegativeFallback
                            || si.totalCost < bestNonNegativeFallback.c
                                - (conservativeEnvelope ? 0.0 : EPS))) {
                    bestNonNegativeFallback = {
                        si.members, si.boundVol,
                        si.loadLower, si.loadUpper, si.totalCost};
                    hasNonNegativeFallback = true;
                }
                continue;
            }
            items.push_back({
                std::move(si.members), si.boundVol,
                si.loadLower, si.loadUpper, si.totalCost});
        }
        // Sort negative costs by the exact binary64 c/q order required by the
        // fractional-knapsack bound.  Nonnegative groups (retained only for a
        // negative SR3 bonus) follow them and are ignored by that relaxation.
        std::sort(items.begin(), items.end(), [](const ItemView& x, const ItemView& y) {
            return pricing_detail::negativeCostDensityLess(
                x.c, x.q, y.c, y.q);
        });

        std::vector<PricingKernelEntry> bestK;
        auto cutoff = [&]() -> double {
            return (int)bestK.size() < topK ? INF : bestK.back().itemCost;
        };

        std::vector<int> current = forced;
        bool pricingInterrupted = false;
        double terminalEnvelope = INF;
        bool frontierArithmeticValid = true;
        const bool preserveFrontier = conservativeEnvelope && params_.forwardUbMode;
        unsigned long long pricingVisits = 0;
        // Track which items are "forbidden" by separate-pair constraints during DFS
        std::vector<int> forbidCount(activeN_, 0);
        // Initialize forbidCount from forced items
        for (int fa : forced) {
            for (int partner : separatePartners[fa]) ++forbidCount[partner];
        }

        auto dfs = [&](auto&& self, int pos, double load,
                       double capacityLoadLower, double capacityLoadUpper,
                       double itemCost,
                       double pendingNegativeBonus) -> void {
            if (pricingInterrupted && !preserveFrontier) return;
            ++pricingVisits;
            // All modes obey the global budget inside one exponential DFS.
            // Only forward mode has an additional per-pricing time slice.
            // Backward interruption never produces a frontier certificate.
            if (((pricingVisits - 1) & 1023ULL) == 0) {
                if (globalDeadlineExpired()) {
                    pricingInterrupted = true;
                    result.globalDeadlineExpired = true;
                } else if (params_.forwardUbMode
                        && std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - pricingStarted).count()
                            >= params_.forwardPricingTimeSliceSec) {
                    pricingInterrupted = true;
                }
            }
#ifdef STAGE2BP_TESTING
            if (params_.forwardUbMode && params_.forwardPricingVisitLimit > 0
                    && pricingVisits >= params_.forwardPricingVisitLimit)
                pricingInterrupted = true;
#endif
            const double remain = conservativeEnvelope
                ? pricing_detail::subtractUp(in_.Qv[v], load)
                : in_.Qv[v] - load;
            double continuation = 0.0;
            if (conservativeEnvelope) {
                continuation =
                    pricing_detail::fractionalContinuationLowerBound(
                        items, pos, remain, pendingNegativeBonus);
            } else {
                // Fast search pass: certification never relies on this
                // floating estimate.  If it misses a marginal column, the
                // subsequent conservative full-domain pass finds it or repairs
                // the dual.  Avoiding nextafter in every DFS suffix keeps the
                // ordinary column-generation loop inexpensive.
                double cap = std::max(0.0, remain) + EPS;
                continuation = pendingNegativeBonus;
                for (int index = pos;
                        index < static_cast<int>(items.size()); ++index) {
                    const double q = items[index].q;
                    const double c = items[index].c;
                    if (c >= 0.0) continue;
                    if (q == 0.0) {
                        continuation += c;
                    } else if (cap <= 0.0) {
                        continue;
                    } else if (q <= cap) {
                        continuation += c;
                        cap -= q;
                    } else {
                        continuation += c * (cap / q);
                        break;
                    }
                }
            }
            const double lb = conservativeEnvelope
                ? pricing_detail::addDown(itemCost, continuation)
                : itemCost + continuation;
            if (preserveFrontier && !std::isfinite(lb)) frontierArithmeticValid = false;
            if (pricingInterrupted || lb >= cutoff() - (conservativeEnvelope ? 0.0 : EPS)) {
                if (preserveFrontier) terminalEnvelope = std::min(terminalEnvelope, lb);
                return;
            }

            if (pos == (int)items.size()) {
                if (current.empty()) return;
                Pattern candidate;
                candidate.v = v;
                candidate.items = current;
                std::sort(candidate.items.begin(), candidate.items.end());
                candidate.items.erase(
                    std::unique(candidate.items.begin(), candidate.items.end()),
                    candidate.items.end());
                candidate.load = patternLoad(candidate.items);
                if (!patternAllowed(candidate, bs)) return;
                // This envelope covers nonempty feasible patterns. Empty and
                // infeasible leaves have no such pattern; the empty pattern
                // is evaluated separately using its actual activation cost.
                if (preserveFrontier) terminalEnvelope = std::min(terminalEnvelope, lb);
                if (itemCost < cutoff()
                        - (conservativeEnvelope ? 0.0 : EPS)) {
                    PricingKernelEntry entry{itemCost, current};
                    auto it = std::lower_bound(bestK.begin(), bestK.end(), entry,
                        [](const PricingKernelEntry& a, const PricingKernelEntry& b) {
                            return a.itemCost < b.itemCost;
                        });
                    bestK.insert(it, entry);
                    if ((int)bestK.size() > topK) bestK.pop_back();
                }
                return;
            }

            const ItemView& it = items[pos];
            // Check if any member of this super-item is forbidden by separate-pairs
            bool forbidden = false;
            for (int a : it.members) {
                if (forbidCount[a] > 0) { forbidden = true; break; }
            }

            bool exactCapacityFeasible = false;
            double childCapacityLoadLower = capacityLoadLower;
            double childCapacityLoadUpper = capacityLoadUpper;
            if (!forbidden) {
                if (!conservativeEnvelope) {
                    childCapacityLoadLower = load + it.q;
                    childCapacityLoadUpper = childCapacityLoadLower;
                    exactCapacityFeasible =
                        childCapacityLoadLower <= in_.Qv[v] + EPS;
                } else if (integerVolumeSumsExact_) {
                    childCapacityLoadLower += it.q;
                    childCapacityLoadUpper = childCapacityLoadLower;
                    exactCapacityFeasible =
                        childCapacityLoadLower <= in_.Qv[v];
                } else {
                    childCapacityLoadLower = std::max(
                        0.0, pricing_detail::addDown(
                            capacityLoadLower, it.loadLower));
                    childCapacityLoadUpper = pricing_detail::addUp(
                        capacityLoadUpper, it.loadUpper);
                    if (childCapacityLoadUpper <= in_.Qv[v]) {
                        exactCapacityFeasible = true;
                    } else if (childCapacityLoadLower > in_.Qv[v]) {
                        exactCapacityFeasible = false;
                    } else {
                        exactCapacityFeasible =
                            exactActiveLoadAtMostWithAdditional(
                                current, it.members, in_.Qv[v]);
                    }
                }
            }
            if (!forbidden && exactCapacityFeasible) {
                // SR3: compute penalty increment for all members
                double sr3Incr = 0.0;
                double childPendingNegativeBonus = pendingNegativeBonus;
                for (int a : it.members) {
                    for (const auto& [ci, p1, p2] : sr3ByCust_[a]) {
                        ++sr3Count[ci];
                        if (sr3Count[ci] == 2) {
                            const double penalty = -rmp.dualSR3[ci];
                            sr3Incr = conservativeEnvelope
                                ? pricing_detail::addDown(sr3Incr, penalty)
                                : sr3Incr + penalty;
                            if (penalty < 0.0) {
                                childPendingNegativeBonus = pricing_detail::addDown(
                                    childPendingNegativeBonus, -penalty);
                            }
                        }
                    }
                    current.push_back(a);
                    // Mark separate-partners as forbidden
                    for (int partner : separatePartners[a]) ++forbidCount[partner];
                }
                const double childItemCost = conservativeEnvelope
                    ? pricing_detail::addDown(
                        pricing_detail::addDown(itemCost, it.c), sr3Incr)
                    : itemCost + it.c + sr3Incr;
                const double childLoad =
                    conservativeEnvelope && !integerVolumeSumsExact_
                    ? std::max(
                        0.0, pricing_detail::addDown(load, it.q))
                    : load + it.q;
                self(self, pos + 1, childLoad,
                     childCapacityLoadLower, childCapacityLoadUpper,
                     childItemCost,
                     childPendingNegativeBonus);
                // Undo
                for (int a : it.members) {
                    current.pop_back();
                    for (const auto& [ci, p1, p2] : sr3ByCust_[a]) --sr3Count[ci];
                    for (int partner : separatePartners[a]) --forbidCount[partner];
                }
            }
            self(self, pos + 1, load,
                 capacityLoadLower, capacityLoadUpper,
                 itemCost, pendingNegativeBonus);
        };

        dfs(dfs, 0,
            conservativeEnvelope ? forcedBoundLoad : forcedLoad,
            forcedLoadLower, forcedLoadUpper,
            forcedCost, pendingNegativeSR3Bonus);

        // Interrupted backward pricing retains feasible columns only. Its
        // exactDomainComplete stays false, its bound stays -INF and its
        // shared kernel stays unready. Forward may separately certify its
        // directed frontier; that existing contract does not apply backward.
        result.globalDeadlineExpired = result.globalDeadlineExpired || globalDeadlineExpired();
        pricingInterrupted = pricingInterrupted || result.globalDeadlineExpired;
        result.forwardTimeSliceExpired = params_.forwardUbMode && pricingInterrupted;

        if (bestK.empty() && forced.empty() && hasNonNegativeFallback) {
            bestK.push_back({bestNonNegativeFallback.c, bestNonNegativeFallback.members});
        }
        if (preserveFrontier && forced.empty() && hasNonNegativeFallback)
            terminalEnvelope = std::min(terminalEnvelope, bestNonNegativeFallback.c);

        // Full conservative pricing must retain the lower envelope of the
        // whole searched/relaxed nonempty domain, not just emitted columns.
        // In particular the cheapest omitted nonnegative group may be rejected
        // by patternAllowed, or cheaper than a negative-item DFS pattern after
        // SR3 penalties. Such a fallback is safe as a LOWER bound regardless;
        // it must never become a feasible column without the usual checks.
        double nonemptyItemCostLowerBound = INF;
        if (conservativeEnvelope && !preserveFrontier) {
            for (const auto& entry : bestK)
                nonemptyItemCostLowerBound = std::min(
                    nonemptyItemCostLowerBound, entry.itemCost);
            if (forced.empty() && hasNonNegativeFallback)
                nonemptyItemCostLowerBound = std::min(
                    nonemptyItemCostLowerBound, bestNonNegativeFallback.c);
        }
        // Partial forward frontiers retain their independent per-vehicle
        // certificate. Share only a completed conservative search, including
        // its pre-filter scalar envelope as well as any feasible K patterns.
        if (kernelCapture && conservativeEnvelope && !pricingInterrupted
                && !preserveFrontier) {
            kernelCapture->ready = true;
            kernelCapture->entries = bestK;
            kernelCapture->nonemptyItemCostLowerBound = nonemptyItemCostLowerBound;
        }

        const double rho_v = rmp.dualVehicle[v];
        bool hasAllowedNonempty = false;
        double bestNonemptyRcLower = conservativeEnvelope && !preserveFrontier
            ? pricing_detail::roundDown(pricing_detail::addDown(
                pricing_detail::addDown(-rho_v, activationCoeff),
                nonemptyItemCostLowerBound))
            : INF;
        for (const auto& entry : bestK) {
            Pattern p;
            p.v = v;
            p.items = entry.items;
            std::sort(p.items.begin(), p.items.end());
            p.items.erase(std::unique(p.items.begin(), p.items.end()), p.items.end());
            p.load = patternLoad(p.items);
            if (!patternAllowed(p, bs)) continue;
            hasAllowedNonempty = true;

            const double rc = conservativeEnvelope
                ? pricing_detail::addDown(
                    pricing_detail::addDown(-rho_v, activationCoeff),
                    entry.itemCost)
                : -rho_v + activationCoeff + entry.itemCost;
            bestNonemptyRcLower = std::min(
                bestNonemptyRcLower,
                conservativeEnvelope ? pricing_detail::roundDown(rc) : rc);
            if (rc < -params_.rcTol) {
                result.columns.push_back({rc, p});
            }
        }
        if (!pricingInterrupted) {
            result.minReducedCostLowerBound = emptyAllowed
                ? std::min(emptyRcLower, bestNonemptyRcLower)
                : bestNonemptyRcLower;
            result.exactDomainComplete = emptyAllowed || hasAllowedNonempty;
        }
        if (preserveFrontier) {
            const double nonemptyEnvelope = pricing_detail::addDown(
                pricing_detail::addDown(-rho_v, activationCoeff), terminalEnvelope);
            result.minReducedCostLowerBound = emptyAllowed
                ? std::min(emptyRcLower, nonemptyEnvelope) : nonemptyEnvelope;
            result.reducedCostEnvelopeCertified =
                frontierArithmeticValid && std::isfinite(nonemptyEnvelope)
                && std::isfinite(result.minReducedCostLowerBound);
        }
        return result;
    }

    PricingResult materializeConservativePricingKernel(
            int v, const BranchState& bs, const RmpResult& rmp,
            const PricingKernelCapture& kernel) const {
        if (!kernel.ready) {
            throw std::logic_error(
                "attempted to materialize an incomplete pricing kernel");
        }

        PricingResult result;
        Pattern emptyPattern;
        emptyPattern.v = v;
        const bool emptyAllowed = patternAllowed(emptyPattern, bs);
        const double emptyRcLower = pricing_detail::roundDown(
            -rmp.dualVehicle[v]);
        const double activationCoeff = activationReducedCostTerm(v, rmp, true);
        const double rho_v = rmp.dualVehicle[v];

        bool hasAllowedNonempty = false;
        double bestNonemptyRcLower = pricing_detail::roundDown(
            pricing_detail::addDown(
                pricing_detail::addDown(-rho_v, activationCoeff),
                kernel.nonemptyItemCostLowerBound));
        for (const auto& entry : kernel.entries) {
            Pattern p;
            p.v = v;
            p.items = entry.items;
            std::sort(p.items.begin(), p.items.end());
            p.items.erase(
                std::unique(p.items.begin(), p.items.end()), p.items.end());
            p.load = patternLoad(p.items);
            if (!patternAllowed(p, bs)) continue;
            hasAllowedNonempty = true;

            const double rc = pricing_detail::addDown(
                pricing_detail::addDown(-rho_v, activationCoeff),
                entry.itemCost);
            bestNonemptyRcLower = std::min(
                bestNonemptyRcLower, pricing_detail::roundDown(rc));
            if (rc < -params_.rcTol) {
                result.columns.push_back({rc, p});
            }
        }
        result.minReducedCostLowerBound = emptyAllowed
            ? std::min(emptyRcLower, bestNonemptyRcLower)
            : bestNonemptyRcLower;
        result.exactDomainComplete = emptyAllowed || hasAllowedNonempty;
        return result;
    }

    static bool sameFiniteDoubleBits(double lhs, double rhs) {
        if (!std::isfinite(lhs) || !std::isfinite(rhs)) return false;
        std::uint64_t lhsBits = 0;
        std::uint64_t rhsBits = 0;
        std::memcpy(&lhsBits, &lhs, sizeof(lhsBits));
        std::memcpy(&rhsBits, &rhs, sizeof(rhsBits));
        return lhsBits == rhsBits;
    }

    bool certificateSearchInputsBitwiseEqual(
            int lhs, int rhs,
            const BranchState& bs,
            const RmpResult& dual) const {
        if (!sameFiniteDoubleBits(in_.Qv[lhs], in_.Qv[rhs])
                || bs.fixY[lhs] != bs.fixY[rhs]
                || bs.fixAlpha[lhs] != bs.fixAlpha[rhs]
                || compatible_[lhs] != compatible_[rhs]) {
            return false;
        }
        for (int a = 0; a < activeN_; ++a) {
            if (!sameFiniteDoubleBits(
                    aggPiAlpha_[lhs][a], aggPiAlpha_[rhs][a])) {
                return false;
            }
        }

        std::vector<int> lhsCover;
        std::vector<int> rhsCover;
        for (int ci = 0; ci < static_cast<int>(coverCuts_.size()); ++ci) {
            if (coverCuts_[ci].vehicle == lhs) lhsCover.push_back(ci);
            if (coverCuts_[ci].vehicle == rhs) rhsCover.push_back(ci);
        }
        if (lhsCover.size() != rhsCover.size()) return false;
        for (std::size_t pos = 0; pos < lhsCover.size(); ++pos) {
            const int li = lhsCover[pos];
            const int ri = rhsCover[pos];
            const auto& lc = coverCuts_[li];
            const auto& rc = coverCuts_[ri];
            if (lc.cover != rc.cover
                    || lc.liftCoeffs != rc.liftCoeffs
                    || !sameFiniteDoubleBits(
                        dual.dualCover[li], dual.dualCover[ri])) {
                return false;
            }
        }
        return true;
    }

    std::vector<int> rawCertificateKernelRepresentatives(
            const BranchState& bs, const RmpResult& dual) const {
        std::vector<int> representatives(in_.m, -1);
        for (int v = 0; v < in_.m; ++v) {
            representatives[v] = v;
            for (int u = 0; u < v; ++u) {
                if (certificateSearchInputsBitwiseEqual(v, u, bs, dual)) {
                    representatives[v] = representatives[u];
                    break;
                }
            }
        }
        return representatives;
    }

    std::vector<PricingResult> priceAllVehiclesForRawCertificate(
            const BranchState& bs, const RmpResult& dual,
            int topK) const {
        const std::vector<int> representatives =
            rawCertificateKernelRepresentatives(bs, dual);
        std::vector<PricingResult> priced(in_.m);
        std::vector<PricingKernelCapture> kernels(in_.m);
        for (int v = 0; v < in_.m; ++v) {
            const int representative = representatives[v];
            if (representative == v) {
                priced[v] = priceVehicle(
                    v, bs, dual, topK, true, &kernels[v]);
            } else if (kernels[representative].ready) {
                priced[v] = materializeConservativePricingKernel(
                    v, bs, dual, kernels[representative]);
            } else {
                // Early/degenerate representative path: retain the complete
                // original oracle instead of inferring missing kernel state.
                priced[v] = priceVehicle(v, bs, dual, topK, true);
            }
        }
        return priced;
    }

    // ===== Heuristic pricing: greedy c/q fill (Optimization #1) =====
    PricingResult heuristicPriceVehicle(int v, const BranchState& bs, const RmpResult& rmp) const {
        PricingResult result;
        if (bs.fixY[v] == 0) return result;

        double activationCoeff = activationReducedCostTerm(v, rmp);

        std::vector<double> itemCoeff(activeN_, 0.0);
        for (int a = 0; a < activeN_; ++a) {
            itemCoeff[a] = -rmp.dualCustomer[a] + aggPiAlpha_[v][a];
        }
        for (int ci = 0; ci < (int)coverCuts_.size(); ++ci) {
            const auto& cc = coverCuts_[ci];
            if (cc.vehicle != v) continue;
            const double sigma = rmp.dualCover[ci];
            if (cc.liftCoeffs.empty()) {
                for (int ca : cc.cover) itemCoeff[ca] -= sigma;
            } else {
                for (int a = 0; a < activeN_; ++a) {
                    if (cc.liftCoeffs[a] > 0) itemCoeff[a] -= sigma * cc.liftCoeffs[a];
                }
            }
        }

        std::vector<int> forced;
        double forcedLoad = 0.0, forcedCost = 0.0;
        struct ItemView { int a; double q; double c; };
        std::vector<ItemView> items;
        for (int a = 0; a < activeN_; ++a) {
            if (bs.fixAlpha[v][a] == 0 || !compatible_[v][a]) continue;
            if (bs.fixAlpha[v][a] == 1) {
                forced.push_back(a);
                forcedLoad += in_.volume[activeOrig_[a]];
                forcedCost += itemCoeff[a];
            } else {
                if (itemCoeff[a] < -1e-12)
                    items.push_back({a, in_.volume[activeOrig_[a]], itemCoeff[a]});
            }
        }

        // SR3 penalty for forced items
        std::vector<int> sr3Count(sr3Cuts_.size(), 0);
        double forcedSR3 = 0.0;
        for (int fa : forced) {
            for (const auto& [ci, p1, p2] : sr3ByCust_[fa]) {
                ++sr3Count[ci];
                if (sr3Count[ci] == 2) forcedSR3 += (-rmp.dualSR3[ci]);
            }
        }
        forcedCost += forcedSR3;

        if (forcedLoad > in_.Qv[v] + EPS) return result;

        // Sort by c/q ascending (most attractive first)
        std::sort(items.begin(), items.end(), [](const ItemView& x, const ItemView& y) {
            return (x.c / std::max(x.q, 1e-12)) < (y.c / std::max(y.q, 1e-12));
        });

        // Greedy pass: pack items respecting capacity and separate-pairs
        auto violatesSeparate = [&](int a, const std::vector<int>& selected) -> bool {
            for (const auto& pr : bs.separatePairs) {
                int other = -1;
                if (pr.first == a) other = pr.second;
                else if (pr.second == a) other = pr.first;
                else continue;
                for (int s : selected) {
                    if (s == other) return true;
                }
            }
            return false;
        };

        auto greedyFill = [&](int skipFirst) -> PricingColumn {
            std::vector<int> selected = forced;
            double load = forcedLoad, cost = forcedCost;
            std::vector<int> localSR3 = sr3Count;

            int skipped = 0;
            for (const auto& it : items) {
                if (skipped < skipFirst) { ++skipped; continue; }
                if (load + it.q > in_.Qv[v] + EPS) continue;
                if (violatesSeparate(it.a, selected)) continue;
                // Together-pair check: if (it.a, b) is a together-pair, b must also be selected
                bool togetherOk = true;
                for (const auto& pr : bs.togetherPairs) {
                    if (pr.first == it.a || pr.second == it.a) {
                        int partner = (pr.first == it.a) ? pr.second : pr.first;
                        bool partnerIn = false;
                        for (int s : selected) { if (s == partner) { partnerIn = true; break; } }
                        if (!partnerIn) { togetherOk = false; break; }
                    }
                }
                if (!togetherOk) continue;

                // SR3 penalty
                double sr3Incr = 0.0;
                for (const auto& [ci, p1, p2] : sr3ByCust_[it.a]) {
                    ++localSR3[ci];
                    if (localSR3[ci] == 2) sr3Incr += (-rmp.dualSR3[ci]);
                }
                selected.push_back(it.a);
                load += it.q;
                cost += it.c + sr3Incr;
            }

            if (selected.empty()) return {INF, {}};
            Pattern p;
            p.v = v;
            p.items = selected;
            std::sort(p.items.begin(), p.items.end());
            p.load = load;
            double rc = -rmp.dualVehicle[v] + activationCoeff + cost;
            return {rc, p};
        };

        // Try primary greedy and a variant skipping the first item for diversity
        for (int skip = 0; skip < std::min(2, (int)items.size() + 1); ++skip) {
            auto col = greedyFill(skip);
            if (col.rc < -params_.rcTol && col.pattern.nonempty()) {
                if (patternAllowed(col.pattern, bs)) {
                    result.columns.push_back(col);
                }
            }
        }
        return result;
    }

    // ===== Vehicle type clustering (Optimization #2) =====
    void computeVehicleTypes(const BranchState& bs, const RmpResult& rmp) {
        typeRep_.clear();
        typeMembers_.clear();

        // Vehicles may share one pricing solve only when their complete
        // reduced-cost functions are identical.  In particular, the fixed
        // nonempty-pattern term includes both the cut activation aggregate
        // and the vehicle-row dual.  Omitting these constants can make the
        // representative have no negative column while another member does.
        struct VehFingerprint {
            double Qv; double activationRcOffset; int fixY;
            std::vector<int> fixAlpha;
            std::vector<double> itemCoeff;
            bool operator==(const VehFingerprint& o) const {
                if (Qv != o.Qv || fixY != o.fixY) return false;
                if (std::fabs(activationRcOffset - o.activationRcOffset) > 1e-10)
                    return false;
                if (fixAlpha != o.fixAlpha) return false;
                if (itemCoeff.size() != o.itemCoeff.size()) return false;
                for (int i = 0; i < (int)itemCoeff.size(); ++i)
                    if (std::fabs(itemCoeff[i] - o.itemCoeff[i]) > 1e-10) return false;
                return true;
            }
        };

        std::vector<VehFingerprint> fps(in_.m);
        for (int v = 0; v < in_.m; ++v) {
            fps[v].Qv = in_.Qv[v];
            fps[v].activationRcOffset = activationReducedCostTerm(v, rmp)
                                        - rmp.dualVehicle[v];
            fps[v].fixY = bs.fixY[v];
            fps[v].fixAlpha = bs.fixAlpha[v];
            fps[v].itemCoeff.resize(activeN_);
            for (int a = 0; a < activeN_; ++a) {
                double c = -rmp.dualCustomer[a] + aggPiAlpha_[v][a];
                for (int ci = 0; ci < (int)coverCuts_.size(); ++ci) {
                    const auto& cc = coverCuts_[ci];
                    if (cc.vehicle != v) continue;
                    if (cc.liftCoeffs.empty()) {
                        for (int ca : cc.cover)
                            if (ca == a) c -= rmp.dualCover[ci];
                    } else {
                        if (cc.liftCoeffs[a] > 0) c -= rmp.dualCover[ci] * cc.liftCoeffs[a];
                    }
                }
                fps[v].itemCoeff[a] = c;
            }
        }

        std::vector<bool> assigned(in_.m, false);
        for (int v = 0; v < in_.m; ++v) {
            if (assigned[v]) continue;
            typeRep_.push_back(v);
            std::vector<int> members = {v};
            assigned[v] = true;
            for (int u = v + 1; u < in_.m; ++u) {
                if (assigned[u]) continue;
                if (fps[v] == fps[u]) {
                    members.push_back(u);
                    assigned[u] = true;
                }
            }
            typeMembers_.push_back(std::move(members));
        }
    }

    // ===== Diving heuristic (Optimization #3) =====
    Solution divingHeuristic(const std::vector<int>& cols, const RmpResult& rmp) const {
        // Repeatedly fix the largest fractional lambda to 1, remove conflicts
        std::vector<double> lam = rmp.lambdaValue;
        std::vector<bool> fixed(cols.size(), false);
        std::vector<bool> removed(cols.size(), false);
        std::vector<int> selectedCols;

        while (true) {
            int bestK = -1;
            double bestVal = 0.0;
            for (int k = 0; k < (int)cols.size(); ++k) {
                if (fixed[k] || removed[k]) continue;
                if (lam[k] > bestVal + 1e-9) {
                    bestVal = lam[k];
                    bestK = k;
                }
            }
            if (bestK < 0 || bestVal < params_.intTol) break;

            // Fix this column
            fixed[bestK] = true;
            selectedCols.push_back(cols[bestK]);
            const Pattern& p = allColumns_[cols[bestK]];

            // Remove conflicting columns (same vehicle or overlapping customers)
            for (int k = 0; k < (int)cols.size(); ++k) {
                if (fixed[k] || removed[k]) continue;
                const Pattern& q = allColumns_[cols[k]];
                if (q.v == p.v) { removed[k] = true; continue; }
                for (int a : p.items) {
                    if (containsActive(q, a)) { removed[k] = true; break; }
                }
            }
        }

        // Build solution from selected columns
        Solution sol;
        sol.feasible = true;
        sol.alpha.assign(in_.m, std::vector<int>(in_.n, 0));
        sol.y.assign(in_.m, 0);
        sol.z.assign(in_.m, 0);
        sol.s.assign(in_.n, 1);
        sol.theta.assign(in_.numSucc, 0.0);

        std::vector<bool> customerCovered(activeN_, false);
        for (int id : selectedCols) {
            const Pattern& p = allColumns_[id];
            if (p.nonempty()) sol.y[p.v] = 1;
            for (int a : p.items) {
                int j = activeOrig_[a];
                sol.alpha[p.v][j] = 1;
                sol.s[j] = 0;
                customerCovered[a] = true;
            }
        }

        // Verify: each vehicle has exactly one pattern (convexity). Some vehicles
        // may not have a selected pattern - they get the empty pattern (valid).
        // Check each active customer is covered at most once
        for (int a = 0; a < activeN_; ++a) {
            int cnt = 0;
            for (int v = 0; v < in_.m; ++v) cnt += sol.alpha[v][activeOrig_[a]];
            if (cnt > 1) { sol.feasible = false; return sol; }
        }

        // Inactive customers are outsourced
        for (int j = 0; j < in_.n; ++j) {
            if (!in_.active[j]) sol.s[j] = 1;
        }

        assignOptimalPurchaseVector(sol);

        // Compute theta from cuts
        for (const Stage3Cut& cut : in_.cuts) {
            double rhs = cut.beta;
            for (int v = 0; v < in_.m; ++v) {
                rhs += cut.piY[v] * sol.y[v];
                for (int j = 0; j < in_.n; ++j) rhs += cut.piAlpha[v][j] * sol.alpha[v][j];
            }
            sol.theta[cut.succ] = std::max(sol.theta[cut.succ], rhs);
        }
        for (double& th : sol.theta) th = std::max(std::max(0.0, params_.thetaLowerBound), th);

        // Compute objective
        double obj = 0.0;
        for (double th : sol.theta) obj += th;
        for (int j = 0; j < in_.n; ++j) obj += in_.cOut[j] * sol.s[j];
        for (int v = 0; v < in_.m; ++v) obj -= in_.piZ[v] * sol.z[v];
        sol.obj = obj;
        return sol;
    }

    // ===== Dual stabilization (Optimization #5) =====
    RmpResult smoothDuals(const RmpResult& rawRmp) {
        if (!stableInitialized_) {
            stableDualVehicle_ = rawRmp.dualVehicle;
            stableDualCustomer_ = rawRmp.dualCustomer;
            stableDualCut_ = rawRmp.dualCut;
            stableDualSR3_ = rawRmp.dualSR3;
            stableDualCover_ = rawRmp.dualCover;
            stableDualActivationLink_ = rawRmp.dualActivationLink;
            stableDualPurchaseConvexity_ = rawRmp.dualPurchaseConvexity;
            stableObj_ = rawRmp.obj;
            stableInitialized_ = true;
            return rawRmp;
        }

        // Update center if LP improved
        if (rawRmp.obj < stableObj_ - EPS) {
            stableDualVehicle_ = rawRmp.dualVehicle;
            stableDualCustomer_ = rawRmp.dualCustomer;
            stableDualCut_ = rawRmp.dualCut;
            stableDualSR3_ = rawRmp.dualSR3;
            stableDualCover_ = rawRmp.dualCover;
            stableDualActivationLink_ = rawRmp.dualActivationLink;
            stableDualPurchaseConvexity_ = rawRmp.dualPurchaseConvexity;
            stableObj_ = rawRmp.obj;
            smoothAlpha_ = std::max(params_.smoothAlphaMin, smoothAlpha_ - 0.05);
        } else {
            smoothAlpha_ = std::min(params_.smoothAlphaMax, smoothAlpha_ + 0.02);
        }

        // Compute smoothed duals
        RmpResult smoothed = rawRmp;
        const double a = smoothAlpha_;
        for (int v = 0; v < in_.m; ++v)
            smoothed.dualVehicle[v] = a * stableDualVehicle_[v] + (1.0 - a) * rawRmp.dualVehicle[v];
        for (int i = 0; i < activeN_; ++i)
            smoothed.dualCustomer[i] = a * stableDualCustomer_[i] + (1.0 - a) * rawRmp.dualCustomer[i];
        for (int r = 0; r < (int)in_.cuts.size(); ++r)
            smoothed.dualCut[r] = a * stableDualCut_[r] + (1.0 - a) * rawRmp.dualCut[r];
        for (int ci = 0; ci < (int)stableDualSR3_.size() && ci < (int)smoothed.dualSR3.size(); ++ci)
            smoothed.dualSR3[ci] = a * stableDualSR3_[ci] + (1.0 - a) * rawRmp.dualSR3[ci];
        for (int ci = 0; ci < (int)stableDualCover_.size() && ci < (int)smoothed.dualCover.size(); ++ci)
            smoothed.dualCover[ci] = a * stableDualCover_[ci] + (1.0 - a) * rawRmp.dualCover[ci];
        for (int v = 0;
                v < static_cast<int>(stableDualActivationLink_.size()); ++v) {
            smoothed.dualActivationLink[v] =
                a * stableDualActivationLink_[v]
                + (1.0 - a) * rawRmp.dualActivationLink[v];
        }
        for (int group = 0;
                group < static_cast<int>(
                    stableDualPurchaseConvexity_.size()); ++group) {
            smoothed.dualPurchaseConvexity[group] =
                a * stableDualPurchaseConvexity_[group]
                + (1.0 - a) * rawRmp.dualPurchaseConvexity[group];
        }
        return smoothed;
    }

    // ===== Column pool culling (Optimization #6) =====
    int cullHighRcColumns(std::vector<int>& cols, RmpResult& rmp) {
        if (params_.cullRcThreshold <= 0.0) return 0;
        int removed = 0;
        // CRITICAL: rmp.lambdaValue is indexed positionally by `cols`. When we
        // drop columns we MUST compact lambdaValue in lockstep, otherwise every
        // downstream consumer (isIntegerRmp, reconstructIntegerSolution,
        // chooseRyanFosterPair, separateCoverCuts, diving) reads mismatched
        // lambda values against the new cols -> wrong incumbent / invalid LB.
        const bool haveLambda = (rmp.lambdaValue.size() == cols.size());
        std::vector<int> kept;
        std::vector<double> keptLambda;
        kept.reserve(cols.size());
        if (haveLambda) keptLambda.reserve(cols.size());
        for (int k = 0; k < (int)cols.size(); ++k) {
            // Approximate RC from lambda value: columns with lambda=0 and large inferred rc
            if (haveLambda && std::fabs(rmp.lambdaValue[k]) < 1e-12) {
                // Compute rc for this column
                const Pattern& p = allColumns_[cols[k]];
                double rc = -rmp.dualVehicle[p.v]
                    + activationReducedCostTerm(p.v, rmp)
                        * (p.nonempty() ? 1.0 : 0.0);
                for (int a : p.items) {
                    rc -= rmp.dualCustomer[a];
                    rc += aggPiAlpha_[p.v][a];
                }
                if (rc > params_.cullRcThreshold) {
                    ++removed;
                    continue;
                }
            }
            kept.push_back(cols[k]);
            if (haveLambda) keptLambda.push_back(rmp.lambdaValue[k]);
        }
        cols.swap(kept);
        if (haveLambda) rmp.lambdaValue.swap(keptLambda);
        return removed;
    }
    void precomputeCutAggregates(const RmpResult& rmp,
                                 bool conservativeEnvelope = false) {
        aggActivation_.assign(in_.m, 0.0);
        aggPiAlpha_.assign(in_.m, std::vector<double>(activeN_, 0.0));
        for (int r = 0; r < (int)in_.cuts.size(); ++r) {
            const double d = rmp.dualCut[r];
            if ((!conservativeEnvelope && std::fabs(d) < 1e-15)
                    || (conservativeEnvelope && d == 0.0)) continue;
            for (int v = 0; v < in_.m; ++v) {
                const double activationTerm = conservativeEnvelope
                    ? pricing_detail::multiplyDown(d, in_.cuts[r].piY[v])
                    : d * in_.cuts[r].piY[v];
                aggActivation_[v] = conservativeEnvelope
                    ? pricing_detail::addDown(
                        aggActivation_[v], activationTerm)
                    : aggActivation_[v] + activationTerm;
                if (conservativeEnvelope) {
                    // The sparse ordinary-pricing index intentionally drops
                    // tiny coefficients.  A proof path may not: even a tiny
                    // negative term can invalidate a reduced-cost certificate.
                    for (int a = 0; a < activeN_; ++a) {
                        const double coeff =
                            in_.cuts[r].piAlpha[v][activeOrig_[a]];
                        if (coeff == 0.0) continue;
                        const double itemTerm =
                            pricing_detail::multiplyDown(d, coeff);
                        aggPiAlpha_[v][a] = pricing_detail::addDown(
                            aggPiAlpha_[v][a], itemTerm);
                    }
                } else {
                    for (const auto& [a, coeff] : cutSparseAlpha_[r][v]) {
                        aggPiAlpha_[v][a] += d * coeff;
                    }
                }
            }
        }
    }

    bool certificateInputsFinite() const {
        if (!std::isfinite(params_.thetaLowerBound)
                || !pricing_detail::finiteVector(in_.volume)
                || !pricing_detail::finiteVector(in_.cOut)
                || !pricing_detail::finiteVector(in_.Qv)
                || !pricing_detail::finiteVector(in_.piZ)) {
            return false;
        }
        for (const Stage3Cut& cut : in_.cuts) {
            if (!std::isfinite(cut.beta)
                    || !pricing_detail::finiteVector(cut.piY)) return false;
            for (const auto& row : cut.piAlpha) {
                if (!pricing_detail::finiteVector(row)) return false;
            }
        }
        return true;
    }

    // Project the raw RMP dual onto all non-column dual constraints before a
    // certificate is attempted.  This is a proof transformation, not a
    // tolerance-based acceptance rule:
    //   Benders/clique >= rows -> dual >= 0;
    //   SR3/cover <= rows      -> dual <= 0;
    //   o_a >= 0              -> mu_a + sum cliqueDual <= cOut_a;
    //   theta_h >= L          -> sum_{r->h} cutDual_r <= 1.
    // Any finite wrong-sign inequality dual is clamped to zero.  Theta and
    // customer constraints are repaired conservatively, and pricing is always
    // repeated under the repaired vector.  Nonfinite data is never certified.
    bool repairRawDualForCertificate(
            RmpResult& dual,
            pricing_detail::InequalityDualRepairStats& signStats,
            std::vector<double>& thetaRcLower,
            std::string& failureReason) const {
        const auto wrongSize = [&]() {
            return static_cast<int>(dual.dualVehicle.size()) != in_.m
                || static_cast<int>(dual.dualCustomer.size()) != activeN_
                || dual.dualCut.size() != in_.cuts.size()
                || dual.dualSR3.size() != sr3Cuts_.size()
                || dual.dualCover.size() != coverCuts_.size()
                || dual.dualClique.size() != cliqueCuts_.size()
                || (params_.usePurchaseOrder
                    && (dual.dualActivationLink.size()
                            != static_cast<std::size_t>(in_.m)
                        || dual.dualPurchaseConvexity.size()
                            != purchaseGroups_.size()))
                || (!params_.usePurchaseOrder
                    && (!dual.dualActivationLink.empty()
                        || !dual.dualPurchaseConvexity.empty()));
        };
        if (wrongSize() || !certificateInputsFinite()
                || !pricing_detail::finiteVector(dual.dualVehicle)
                || !pricing_detail::finiteVector(dual.dualCustomer)) {
            failureReason = "nonfinite_or_malformed_dual";
            return false;
        }

        for (double& value : dual.dualCut)
            pricing_detail::repairDualNonnegative(value, signStats);
        for (int ci = 0; ci < static_cast<int>(dual.dualSR3.size()); ++ci) {
            if (!sr3Cuts_[ci].active) dual.dualSR3[ci] = 0.0;
            pricing_detail::repairDualNonpositive(dual.dualSR3[ci], signStats);
        }
        for (int ci = 0; ci < static_cast<int>(dual.dualCover.size()); ++ci) {
            if (!coverCuts_[ci].active) dual.dualCover[ci] = 0.0;
            pricing_detail::repairDualNonpositive(dual.dualCover[ci], signStats);
        }
        for (double& value : dual.dualClique)
            pricing_detail::repairDualNonnegative(value, signStats);
        for (double& value : dual.dualActivationLink)
            pricing_detail::repairDualNonpositive(value, signStats);
        if (!pricing_detail::finiteVector(dual.dualPurchaseConvexity))
            signStats.finite = false;
        if (!signStats.finite) {
            failureReason = "nonfinite_inequality_dual";
            return false;
        }

        // A theta lower bound at Gurobi's -infinity sentinel makes theta free,
        // so the corresponding dual constraint must be equality.  A simple,
        // always-feasible projection is to put unit mass on one cut for that
        // successor and zero on the rest; the subsequent complete pricing pass
        // accounts for the potentially large change.
        const bool thetaIsFree = params_.thetaLowerBound <= -GRB_INFINITY;
        thetaRcLower.assign(in_.numSucc, 0.0);
        for (int h = 0; h < in_.numSucc; ++h) {
            std::vector<int> rows;
            for (int r = 0; r < static_cast<int>(in_.cuts.size()); ++r) {
                if (in_.cuts[r].succ == h) rows.push_back(r);
            }
            if (thetaIsFree) {
                if (rows.empty()) {
                    failureReason = "free_theta_without_supporting_cut";
                    return false;
                }
                for (int r : rows) dual.dualCut[r] = 0.0;
                dual.dualCut[rows.front()] = 1.0;
                thetaRcLower[h] = 0.0;
                continue;
            }

            // Keep a full ulp of room below one.  addUp then provides an upper
            // envelope on the mathematical sum of the binary64 dual values.
            const double target = std::nextafter(1.0, 0.0);
            for (int attempt = 0; attempt < 8; ++attempt) {
                double sumUpper = 0.0;
                for (int r : rows) {
                    if (dual.dualCut[r] > 0.0)
                        sumUpper = pricing_detail::addUp(
                            sumUpper, dual.dualCut[r]);
                }
                if (!std::isfinite(sumUpper)) {
                    failureReason = "nonfinite_theta_dual_sum";
                    return false;
                }
                if (sumUpper <= target) break;
                double factor = pricing_detail::roundDown(target / sumUpper);
                factor = std::max(0.0, std::min(1.0, factor));
                for (int r : rows) {
                    if (dual.dualCut[r] <= 0.0) continue;
                    dual.dualCut[r] = std::max(
                        0.0, pricing_detail::multiplyDown(
                            dual.dualCut[r], factor));
                }
            }
            double sumUpper = 0.0;
            for (int r : rows) {
                if (dual.dualCut[r] > 0.0)
                    sumUpper = pricing_detail::addUp(
                        sumUpper, dual.dualCut[r]);
            }
            if (!std::isfinite(sumUpper) || sumUpper > target) {
                failureReason = "theta_dual_projection_failed";
                return false;
            }
            thetaRcLower[h] = pricing_detail::subtractDown(1.0, sumUpper);
            if (!(thetaRcLower[h] >= 0.0)
                    || !std::isfinite(thetaRcLower[h])) {
                failureReason = "theta_reduced_cost_not_feasible";
                return false;
            }
        }

        // Repair the reduced cost of every outsourcing variable by lowering
        // its free customer-row dual.  This can only increase lambda reduced
        // costs, so it never undermines the subsequent pricing certificate.
        for (int a = 0; a < activeN_; ++a) {
            double cliqueSumUpper = 0.0;
            for (int ci = 0; ci < static_cast<int>(cliqueCuts_.size()); ++ci) {
                if (!std::binary_search(cliqueCuts_[ci].clique.begin(),
                                        cliqueCuts_[ci].clique.end(), a)) continue;
                if (dual.dualClique[ci] > 0.0) {
                    cliqueSumUpper = pricing_detail::addUp(
                        cliqueSumUpper, dual.dualClique[ci]);
                }
            }
            const double allowedMuUpper = pricing_detail::subtractDown(
                in_.cOut[activeOrig_[a]], cliqueSumUpper);
            if (!std::isfinite(allowedMuUpper)) {
                failureReason = "nonfinite_outsource_reduced_cost";
                return false;
            }
            if (dual.dualCustomer[a] > allowedMuUpper)
                dual.dualCustomer[a] = allowedMuUpper;

            const double oRcLower = pricing_detail::subtractDown(
                pricing_detail::subtractDown(
                    in_.cOut[activeOrig_[a]], dual.dualCustomer[a]),
                cliqueSumUpper);
            if (!(oRcLower >= 0.0) || !std::isfinite(oRcLower)) {
                failureReason = "outsource_dual_projection_failed";
                return false;
            }
        }

        // Certify the finite purchase-prefix block.  Each type chooses one
        // nonnegative prefix variable.  Lowering its free convexity-row dual
        // by the most negative prefix reduced cost makes every prefix column
        // dual-feasible and lowers the dual objective by the same amount.
        for (int group = 0;
                group < static_cast<int>(purchaseGroups_.size()); ++group) {
            const auto& vehicles = purchaseGroups_[group];
            double bestRcLower = INF;
            double prefixCostLower = 0.0;
            double linkSumLower = 0.0;
            for (int prefix = 0;
                    prefix <= static_cast<int>(vehicles.size()); ++prefix) {
                if (prefix > 0) {
                    const int vehicle = vehicles[prefix - 1];
                    prefixCostLower = pricing_detail::addDown(
                        prefixCostLower, -in_.piZ[vehicle]);
                    linkSumLower = pricing_detail::addDown(
                        linkSumLower, dual.dualActivationLink[vehicle]);
                }
                const double rcLower = pricing_detail::addDown(
                    pricing_detail::subtractDown(
                        prefixCostLower,
                        dual.dualPurchaseConvexity[group]),
                    linkSumLower);
                bestRcLower = std::min(bestRcLower, rcLower);
            }
            if (!std::isfinite(bestRcLower)) {
                failureReason = "nonfinite_purchase_prefix_reduced_cost";
                return false;
            }
            if (bestRcLower < 0.0) {
                double reduction = pricing_detail::roundUp(-bestRcLower);
                reduction = pricing_detail::addUp(
                    reduction,
                    pricing_detail::dualSignNoiseSlack(
                        std::max(
                            std::fabs(bestRcLower),
                            std::fabs(
                                dual.dualPurchaseConvexity[group]))));
                double repaired = pricing_detail::subtractDown(
                    dual.dualPurchaseConvexity[group], reduction);
                if (!std::isfinite(repaired)) {
                    failureReason = "purchase_convexity_dual_projection_failed";
                    return false;
                }
                dual.dualPurchaseConvexity[group] = repaired;
            }
        }
        return true;
    }

    // Explicit dual objective of the repaired full-master dual.  The RMP
    // primal ObjVal is intentionally absent.  With L=thetaLowerBound:
    //   D = constants + sum_v rho_v + sum_a mu_a
    //       + sum_r beta_r d_r
    //       + sum_SR3 1*sigma + sum_cover (|C|-1)*tau
    //       + sum_clique (|K|-m)*gamma
    //       + sum_h L * (1-sum_{r->h} d_r).
    // Every product and accumulation is rounded toward -infinity.
    bool explicitRepairedDualObjective(
            const RmpResult& dual,
            const std::vector<double>& thetaRcLower,
            double& objective,
            std::string& failureReason) const {
        double value = 0.0;
        if (params_.usePurchaseOrder) {
            for (double rho : dual.dualPurchaseConvexity)
                value = pricing_detail::addDown(value, rho);
        } else {
            for (int v = 0; v < in_.m; ++v) {
                value = pricing_detail::addDown(
                    value, -std::max(in_.piZ[v], 0.0));
            }
        }
        for (int j = 0; j < in_.n; ++j) {
            if (origToActive_[j] < 0)
                value = pricing_detail::addDown(value, in_.cOut[j]);
        }
        for (double rho : dual.dualVehicle)
            value = pricing_detail::addDown(value, rho);
        for (double mu : dual.dualCustomer)
            value = pricing_detail::addDown(value, mu);
        for (int r = 0; r < static_cast<int>(in_.cuts.size()); ++r) {
            value = pricing_detail::addDown(
                value, pricing_detail::multiplyDown(
                    in_.cuts[r].beta, dual.dualCut[r]));
        }
        for (int ci = 0; ci < static_cast<int>(sr3Cuts_.size()); ++ci) {
            if (sr3Cuts_[ci].active)
                value = pricing_detail::addDown(value, dual.dualSR3[ci]);
        }
        for (int ci = 0; ci < static_cast<int>(coverCuts_.size()); ++ci) {
            if (!coverCuts_[ci].active) continue;
            value = pricing_detail::addDown(
                value, pricing_detail::multiplyDown(
                    static_cast<double>(coverCuts_[ci].cover.size() - 1),
                    dual.dualCover[ci]));
        }
        for (int ci = 0; ci < static_cast<int>(cliqueCuts_.size()); ++ci) {
            value = pricing_detail::addDown(
                value, pricing_detail::multiplyDown(
                    static_cast<double>(cliqueCuts_[ci].clique.size() - in_.m),
                    dual.dualClique[ci]));
        }
        if (params_.thetaLowerBound > -GRB_INFINITY) {
            if (thetaRcLower.size() != static_cast<std::size_t>(in_.numSucc)) {
                failureReason = "theta_reduced_cost_size_mismatch";
                return false;
            }
            if (params_.thetaLowerBound != 0.0) {
                for (int h = 0; h < in_.numSucc; ++h) {
                    double rcForProduct = thetaRcLower[h];
                    if (params_.thetaLowerBound < 0.0) {
                        // L < 0 reverses the interval direction: multiplying
                        // L by a lower envelope on rc would give an *upper*
                        // envelope on L*rc.  Build an upper rc envelope from a
                        // downward sum of the repaired nonnegative d values.
                        double cutDualSumLower = 0.0;
                        bool hasSupportingCut = false;
                        for (int r = 0;
                                r < static_cast<int>(in_.cuts.size()); ++r) {
                            if (in_.cuts[r].succ != h
                                    || dual.dualCut[r] == 0.0) continue;
                            hasSupportingCut = true;
                            cutDualSumLower = pricing_detail::addDown(
                                cutDualSumLower, dual.dualCut[r]);
                        }
                        rcForProduct = hasSupportingCut
                            ? pricing_detail::subtractUp(
                                1.0, cutDualSumLower)
                            : 1.0;
                        if (!(rcForProduct >= 0.0)
                                || !std::isfinite(rcForProduct)) {
                            failureReason =
                                "theta_reduced_cost_upper_envelope_failed";
                            return false;
                        }
                    }
                    value = pricing_detail::addDown(
                        value, pricing_detail::multiplyDown(
                            params_.thetaLowerBound, rcForProduct));
                }
            }
        }
        if (!std::isfinite(value)) {
            failureReason = "nonfinite_explicit_dual_objective";
            return false;
        }
        objective = pricing_detail::roundDown(value);
        return true;
    }

    DualCertificationAttempt certifyRawDual(
            const BranchState& bs,
            const RmpResult& rawRmp,
            std::vector<int>& cols,
            int topK) {
        DualCertificationAttempt attempt;
        RmpResult dual = rawRmp;
        std::vector<double> thetaRcLower;
        if (!repairRawDualForCertificate(
                dual, attempt.signRepairs, thetaRcLower,
                attempt.failureReason)) {
            return attempt;
        }

        // Recompute every aggregate using directed lower arithmetic, then run
        // the complete non-heuristic oracle for every vehicle.  Vehicles with
        // a bitwise-identical combinatorial search key may share only the DFS
        // kernel; columns, reduced costs, minimum-RC envelopes, and repairs
        // remain independently materialized for every vehicle.
        precomputeCutAggregates(dual, true);
        std::vector<PricingResult> priced =
            priceAllVehiclesForRawCertificate(bs, dual, topK);
        attempt.pricingComplete = true;
        for (const auto& result : priced)
            if (result.globalDeadlineExpired) ++pricing_deadline_interruptions_;
        for (int v = 0; v < in_.m; ++v) {
            attempt.pricingComplete = attempt.pricingComplete && priced[v].exactDomainComplete;
            if (!(priced[v].exactDomainComplete
                    || (params_.forwardUbMode && priced[v].reducedCostEnvelopeCertified))
                    || !std::isfinite(priced[v].minReducedCostLowerBound)) {
                attempt.pricingComplete = false;
                attempt.failureReason = "incomplete_or_nonfinite_exact_pricing";
                return attempt;
            }
            for (const auto& column : priced[v].columns) {
                const int id = addColumnIfNew(column.pattern);
                if (!isInCols(id)) {
                    cols.push_back(id);
                    markInCols(id);
                    ++attempt.newColumns;
                }
            }
        }
        if (attempt.newColumns > 0 && !params_.forwardUbMode) return attempt;

        // Repair each convexity-row dual using a lower envelope L_v <= min RC.
        // Lowering rho_v by epsilon_v raises every vehicle-v column reduced
        // cost by exactly epsilon_v and lowers the dual objective by the same
        // amount because the convexity RHS is one.  The directed algebraic
        // recheck below is therefore a complete recheck of the column domain;
        // it does not assume the best floating DFS incumbent is exact.
        for (int v = 0; v < in_.m; ++v) {
            const double lower = priced[v].minReducedCostLowerBound;
            if (lower >= 0.0) continue;
            double epsilon = pricing_detail::roundUp(-lower);
            epsilon = pricing_detail::addUp(
                epsilon,
                pricing_detail::dualSignNoiseSlack(
                    std::max(std::fabs(lower),
                             std::fabs(dual.dualVehicle[v]))));
            for (int guard = 0; guard < 8
                    && pricing_detail::addDown(lower, epsilon) < 0.0;
                    ++guard) {
                epsilon = std::nextafter(
                    epsilon, std::numeric_limits<double>::infinity());
            }
            double repairedRho = pricing_detail::subtractDown(
                dual.dualVehicle[v], epsilon);
            double reductionLower = pricing_detail::subtractDown(
                dual.dualVehicle[v], repairedRho);
            for (int guard = 0; guard < 8
                    && pricing_detail::addDown(lower, reductionLower) < 0.0;
                    ++guard) {
                repairedRho = std::nextafter(
                    repairedRho, -std::numeric_limits<double>::infinity());
                reductionLower = pricing_detail::subtractDown(
                    dual.dualVehicle[v], repairedRho);
            }
            if (!std::isfinite(repairedRho)
                    || pricing_detail::addDown(
                        lower, reductionLower) < 0.0) {
                attempt.failureReason = "convexity_dual_projection_failed";
                return attempt;
            }
            dual.dualVehicle[v] = repairedRho;
        }

        if (!explicitRepairedDualObjective(
                dual, thetaRcLower, attempt.bound,
                attempt.failureReason)) {
            return attempt;
        }
        attempt.certified = true;
        return attempt;
    }

    // by strong-branching trial evaluations (short cap for ranking only).
    //   feasibleOut = true iff at least one RMP LP solve was feasible. In that
    //                 case `finalRmp` holds the last feasible RMP (possibly
    //                 partial if maxIters was hit).
    //   Returns true iff CG converged (pricing found no new column) -- then
    //                 finalRmp.obj is a valid LP relaxation lower bound.
    //   On wall-clock timeout sets timed_out_=true and returns false.
    //   Hitting maxIters without convergence is NOT a timeout here -- caller
    //   decides what to do (the main loop treats it as a soft timeout; strong-
    //   branching trials are happy with the partial RMP as a ranking signal).
    bool columnGenerationCore(const BranchState& bs, std::vector<int>& cols,
                              RmpResult& finalRmp, int maxIters, bool& feasibleOut,
                              int depth = 0) {
        feasibleOut = false;
        // Optimistic: every certified-convergence return path below keeps this
        // true. Only the forward heuristic-only fast path clears it.
        last_cg_certified_ = true;
        ensureBasicColumns(bs, cols);

        // Reset dual stabilization for this CG run
        stableInitialized_ = false;
        smoothAlpha_ = params_.smoothAlphaInit;
        firstLpSolveInCG_ = true;

        // Adaptive pricingTopK based on depth
        int effectiveTopK = params_.pricingTopK;
        if (depth == 0) {
            effectiveTopK = params_.pricingTopKRoot;
        } else if (depth <= params_.adaptiveTopKShallowDepth) {
            effectiveTopK = params_.pricingTopKShallow;
        } else {
            effectiveTopK = params_.pricingTopKDeep;
        }

        for (int iter = 0; iter < maxIters; ++iter) {
            if (backwardTargetSatisfied()) return false;
            if (depth != 999) improveForwardDuringCg(bs, cols, false, false, &finalRmp);
            if (forward_reference_stop_) {
                last_cg_certified_ = false;
                return false;
            }
            if (forwardRootDualBound(bs, cols, finalRmp, depth))
                return !forward_reference_stop_;
            if (checkTimeout()) {
                finalRmp = RmpResult{};
                return false;
            }
            RmpResult rmp = solveRmpLp(cols);
            firstLpSolveInCG_ = false;
            if (!rmp.feasible) {
                finalRmp = rmp;
                return false;
            }
            feasibleOut = true;
            finalRmp = rmp;

            // Dual stabilization (Wentges): smooth duals for pricing
            RmpResult pricingRmp = rmp;
            if (params_.useDualStabilization) {
                pricingRmp = smoothDuals(rmp);
            }

            ++cg_iters_total_;
            if (depth == 0) ++root_cg_iters_;
            int added = 0;
            double bestRc = 0.0;
            bool pricingYielded = false;
            const auto t_price0 = std::chrono::steady_clock::now();

            precomputeCutAggregates(pricingRmp);

            // Helper to add pricing results
            auto addPricingResults = [&](const PricingResult& pr) {
                if (pr.globalDeadlineExpired) ++pricing_deadline_interruptions_;
                pricingYielded = pricingYielded || pr.forwardTimeSliceExpired;
                for (const auto& col : pr.columns) {
                    const int id = addColumnIfNew(col.pattern);
                    if (!isInCols(id)) {
                        cols.push_back(id);
                        markInCols(id);
                        ++added;
                        bestRc = std::min(bestRc, col.rc);
                    }
                }
            };

            // Phase 1: Heuristic pricing (fast, O(n log n) per vehicle)
            if (params_.useHeuristicPricing) {
                if (params_.useVehicleClustering) {
                    computeVehicleTypes(bs, pricingRmp);
                    for (int t = 0; t < (int)typeRep_.size(); ++t) {
                        int rep = typeRep_[t];
                        PricingResult pr = heuristicPriceVehicle(rep, bs, pricingRmp);
                        addPricingResults(pr);
                        for (int v : typeMembers_[t]) {
                            if (v == rep) continue;
                            for (const auto& col : pr.columns) {
                                Pattern cloned = col.pattern;
                                cloned.v = v;
                                if (patternAllowed(cloned, bs) && cloned.load <= in_.Qv[v] + EPS) {
                                    double rc = -pricingRmp.dualVehicle[v]
                                        + activationReducedCostTerm(v, pricingRmp)
                                            * (cloned.nonempty() ? 1.0 : 0.0);
                                    for (int a : cloned.items) {
                                        rc -= pricingRmp.dualCustomer[a];
                                        rc += aggPiAlpha_[v][a];
                                    }
                                    if (rc < -params_.rcTol) {
                                        PricingResult clonePr;
                                        clonePr.columns.push_back({rc, cloned});
                                        addPricingResults(clonePr);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    for (int v = 0; v < in_.m; ++v) {
                        PricingResult pr = heuristicPriceVehicle(v, bs, pricingRmp);
                        addPricingResults(pr);
                    }
                }

                if (added > 0) {
                    pricing_time_s_ += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t_price0).count();
                    if (params_.verbose) {
                        std::cerr << "  CG iter " << iter
                                  << " obj=" << std::setprecision(12) << rmp.obj
                                  << " cols=" << cols.size()
                                  << " added=" << added << " (heuristic)"
                                  << " bestRc=" << bestRc << "\n";
                    }
                    ++heuristic_pricing_successes_;
                    continue;  // skip exact DFS this iteration
                }
            }

            // Forward UB fast path (depth>0): heuristic pricing found no
            // improving column. Skip the (expensive) exact DFS that would only
            // serve to CERTIFY the LP bound -- forward needs a good incumbent,
            // not an optimality proof. The RMP obj here is a subset-of-columns
            // value (>= true LP optimum), hence NOT a valid lower bound, so we
            // flag the run uncertified; branchAndPrice will keep the parent's
            // valid inherited_lb instead of using rmp.obj.
            if (params_.forwardUbMode && params_.useHeuristicPricing && depth > 0) {
                last_cg_certified_ = false;
                return true;
            }

            // Phase 2: Exact DFS pricing (only when heuristic found nothing)
            ++exact_pricing_calls_;
            added = 0;
            bestRc = 0.0;

            if (params_.useVehicleClustering) {
                if (!params_.useHeuristicPricing)
                    computeVehicleTypes(bs, pricingRmp);

                // Parallel pricing across vehicle types
                std::vector<PricingResult> typeResults(typeRep_.size());
#ifdef _OPENMP
                #pragma omp parallel for num_threads(params_.numThreads) schedule(dynamic) if(params_.numThreads > 1)
#endif
                for (int t = 0; t < (int)typeRep_.size(); ++t) {
                    typeResults[t] = priceVehicle(typeRep_[t], bs, pricingRmp, effectiveTopK);
                }

                // Serial merge
                for (int t = 0; t < (int)typeRep_.size(); ++t) {
                    addPricingResults(typeResults[t]);
                    for (int v : typeMembers_[t]) {
                        if (v == typeRep_[t]) continue;
                        for (const auto& col : typeResults[t].columns) {
                            Pattern cloned = col.pattern;
                            cloned.v = v;
                            if (patternAllowed(cloned, bs) && cloned.load <= in_.Qv[v] + EPS) {
                                double rc = -pricingRmp.dualVehicle[v]
                                    + activationReducedCostTerm(v, pricingRmp)
                                        * (cloned.nonempty() ? 1.0 : 0.0);
                                for (int a : cloned.items) {
                                    rc -= pricingRmp.dualCustomer[a];
                                    rc += aggPiAlpha_[v][a];
                                }
                                if (rc < -params_.rcTol) {
                                    PricingResult clonePr;
                                    clonePr.columns.push_back({rc, cloned});
                                    addPricingResults(clonePr);
                                }
                            }
                        }
                    }
                }
            } else {
                // Parallel pricing across all vehicles
                std::vector<PricingResult> vehResults(in_.m);
#ifdef _OPENMP
                #pragma omp parallel for num_threads(params_.numThreads) schedule(dynamic) if(params_.numThreads > 1)
#endif
                for (int v = 0; v < in_.m; ++v) {
                    vehResults[v] = priceVehicle(v, bs, pricingRmp, effectiveTopK);
                }
                for (int v = 0; v < in_.m; ++v) {
                    addPricingResults(vehResults[v]);
                }
            }

            pricing_time_s_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_price0).count();

            // OpenMP pricing results are merged on this thread. Expiration
            // cannot promote a partial RMP or an incomplete DFS into a bound.
            if (checkTimeout()) {
                last_cg_certified_ = false;
                finalRmp = RmpResult{};
                return false;
            }

            if (pricingYielded) {
                improveForwardDuringCg(bs, cols, false, true, &finalRmp);
                if (forward_reference_stop_) {
                    last_cg_certified_ = false;
                    return false;
                }
                if (added == 0) {
                    // Same uncertified forward return already used at child
                    // nodes after heuristic pricing. Continue primal search
                    // and the existing branch tree with inherited LB only.
                    last_cg_certified_ = false;
                    finalRmp.dualBoundCertified = false;
                    finalRmp.certifiedDualBound = -INF;
                    return true;
                }
            }

            if (params_.verbose) {
                std::cerr << "  CG iter " << iter
                          << " obj=" << std::setprecision(12) << rmp.obj
                          << " cols=" << cols.size()
                          << " added=" << added
                          << " bestRc=" << bestRc << "\n";
            }

            if (added == 0) {
                // Certification is deliberately separate from ordinary
                // pricing.  It projects/checks the raw dual, re-runs the exact
                // oracle for every vehicle (no clustering or heuristic), and
                // computes an explicit feasible-dual objective.  The primal
                // RMP ObjVal is never relabelled as a mathematical lower bound.
                const auto t_cert0 = std::chrono::steady_clock::now();
                DualCertificationAttempt certificate =
                    certifyRawDual(bs, rmp, cols, effectiveTopK);
                pricing_time_s_ += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t_cert0).count();
                if (checkTimeout()) {
                    last_cg_certified_ = false;
                    finalRmp = RmpResult{};
                    return false;
                }
                if (certificate.newColumns > 0) continue;
                if (certificate.certified && certificate.pricingComplete) {
                    finalRmp.dualBoundCertified = true;
                    finalRmp.certifiedDualBound = certificate.bound;
                    last_cg_certified_ = true;
                } else {
                    finalRmp.dualBoundCertified = false;
                    finalRmp.certifiedDualBound = -INF;
                    last_cg_certified_ = false;
                    if (params_.forwardUbMode)
                        improveForwardDuringCg(bs, cols, false, true, &finalRmp);
                    if (params_.verbose) {
                        std::cerr << "  [stage2bp] raw dual uncertified: "
                                  << certificate.failureReason << "\n";
                    }
                }
                return true;
            }
        }
        if (depth != 999) improveForwardDuringCg(bs, cols, true, false, &finalRmp);
        return false;  // hit maxIters without convergence
    }

    bool columnGeneration(const BranchState& bs, std::vector<int>& cols,
                          RmpResult& finalRmp, int depth = 0) {
        bool feasible = false;
        const bool converged = columnGenerationCore(bs, cols, finalRmp,
                                                    params_.maxColgenIters, feasible, depth);
        if (forward_reference_stop_ || backward_gap_stop_) return false;
        if (timed_out_) {
            finalRmp = RmpResult{};
            return false;
        }
        if (!feasible) {
            // First RMP LP was infeasible -- propagate (finalRmp.feasible=false).
            return false;
        }
        if (!converged) {
            // Hit maxColgenIters without convergence: treat as soft timeout.
            // The RMP obj is NOT a valid LP-LB at this point.
            timed_out_ = true;
            finalRmp = RmpResult{};
            return false;
        }
        return true;
    }

    bool isIntegerRmp(const RmpResult& rmp) const {
        for (double v : rmp.lambdaValue) {
            if (std::fabs(v - std::round(v)) > params_.intTol) return false;
        }
        return true;
    }

    std::pair<int,int> chooseRyanFosterPair(const RmpResult& rmp, const std::vector<int>& cols,
                                            const BranchState& bs) const {
        // x_ab = sum_k lambda_k * 1{a,b in pattern_k}. Branch when x_ab is fractional.
        std::vector<std::vector<double>> xab(activeN_, std::vector<double>(activeN_, 0.0));
        for (int k = 0; k < (int)cols.size(); ++k) {
            const double lam = rmp.lambdaValue[k];
            if (lam <= params_.intTol) continue;
            const Pattern& p = allColumns_[cols[k]];
            const auto& its = p.items;
            for (int i = 0; i < (int)its.size(); ++i) {
                for (int j = i + 1; j < (int)its.size(); ++j) {
                    const int a = its[i], b = its[j];
                    xab[a][b] += lam;
                }
            }
        }

        auto isAlreadyBranched = [&](int a, int b) {
            const auto pr = normPair(a, b);
            for (const auto& q : bs.togetherPairs) if (normPair(q.first, q.second) == pr) return true;
            for (const auto& q : bs.separatePairs) if (normPair(q.first, q.second) == pr) return true;
            return false;
        };

        int bestA = -1, bestB = -1;
        double bestScore = INF;
        for (int a = 0; a < activeN_; ++a) {
            for (int b = a + 1; b < activeN_; ++b) {
                if (isAlreadyBranched(a, b)) continue;
                const double x = xab[a][b];
                if (x > params_.intTol && x < 1.0 - params_.intTol) {
                    const double score = std::fabs(x - 0.5);
                    if (score < bestScore) {
                        bestScore = score;
                        bestA = a;
                        bestB = b;
                    }
                }
            }
        }
        return {bestA, bestB};
    }

    std::pair<int,int> chooseAlphaBranch(const RmpResult& rmp, const BranchState& bs) const {
        int bestV = -1, bestA = -1;
        double bestScore = INF;
        for (int v = 0; v < in_.m; ++v) {
            for (int a = 0; a < activeN_; ++a) {
                if (bs.fixAlpha[v][a] != -1) continue;
                const double x = rmp.alphaValue[v][a];
                if (x > params_.intTol && x < 1.0 - params_.intTol) {
                    const double score = std::fabs(x - 0.5);
                    if (score < bestScore) {
                        bestScore = score;
                        bestV = v;
                        bestA = a;
                    }
                }
            }
        }
        return {bestV, bestA};
    }

    std::vector<std::pair<int,int>> topAlphaBranchCandidates(const RmpResult& rmp,
                                                             const BranchState& bs,
                                                             int topK) const {
        struct Cand { double score; int v; int a; };
        std::vector<Cand> cands;
        cands.reserve(in_.m * activeN_);
        for (int v = 0; v < in_.m; ++v) {
            for (int a = 0; a < activeN_; ++a) {
                if (bs.fixAlpha[v][a] != -1) continue;
                const double x = rmp.alphaValue[v][a];
                if (x > params_.intTol && x < 1.0 - params_.intTol) {
                    cands.push_back({std::fabs(x - 0.5), v, a});
                }
            }
        }
        std::sort(cands.begin(), cands.end(), [](const Cand& p, const Cand& q) {
            return p.score < q.score;
        });
        if ((int)cands.size() > topK) cands.resize(topK);
        std::vector<std::pair<int,int>> out;
        out.reserve(cands.size());
        for (const auto& c : cands) out.push_back({c.v, c.a});
        return out;
    }

    // Trial LP evaluation for strong branching. Uses a *capped* CG
    // (strongBranchMaxCG) so the trial cost stays bounded at large n. The
    // returned value is the last RMP objective seen -- not necessarily a
    // proven LP-LB on the child, but a good ranking signal. Strong branching
    // is heuristic in nature (it only picks the branching variable) so this
    // approximation does NOT compromise exactness of the BP search.
    std::pair<double, std::vector<int>> evaluateAlphaBranchLb(const BranchState& bs, const std::vector<int>& cols,
                                 int v, int a, bool oneBranch) {
        BranchState child = bs;
        if (oneBranch) applyBranchAlphaOne(child, v, a);
        else applyBranchAlphaZero(child, v, a);
        if (!quickBranchFeasible(child)) return {INF, {}};

        const int poolBefore = (int)allColumns_.size();
        std::vector<int> colsChild = cols;
        RmpResult rmpChild;
        bool feasible = false;
        const int trialIters = std::max(1, std::min(params_.maxColgenIters,
                                                    params_.strongBranchMaxCG));
        columnGenerationCore(child, colsChild, rmpChild, trialIters, feasible, 999);
        if (timed_out_) return {INF, {}};
        if (!feasible) return {INF, {}};

        std::vector<int> newIds;
        const int poolAfter = (int)allColumns_.size();
        for (int id = poolBefore; id < poolAfter; ++id)
            newIds.push_back(id);
        return {rmpChild.obj, std::move(newIds)};
    }

    std::pair<int,int> chooseAlphaBranchStrong(const RmpResult& rmp, const BranchState& bs,
                                               std::vector<int>& cols, int depth) {
        // Skip strong branching at deep nodes: cost no longer pays off.
        if (depth > params_.strongBranchMaxDepth) return {-1, -1};
        auto cands = topAlphaBranchCandidates(rmp, bs, std::max(1, params_.strongBranchTopK));
        if (cands.empty()) return {-1, -1};
        if ((int)cands.size() == 1) return cands[0];

        int bestV = cands[0].first, bestA = cands[0].second;
        double bestScore = -INF;
        std::vector<int> allNewIds;
        for (const auto& pr : cands) {
            const int v = pr.first, a = pr.second;
            auto [lbOne, newIdsOne] = evaluateAlphaBranchLb(bs, cols, v, a, true);
            if (timed_out_) return {-1, -1};
            allNewIds.insert(allNewIds.end(), newIdsOne.begin(), newIdsOne.end());
            auto [lbZero, newIdsZero] = evaluateAlphaBranchLb(bs, cols, v, a, false);
            if (timed_out_) return {-1, -1};
            allNewIds.insert(allNewIds.end(), newIdsZero.begin(), newIdsZero.end());
            const double score = std::min(lbOne, lbZero);
            if (score > bestScore + 1e-9) {
                bestScore = score;
                bestV = v;
                bestA = a;
            }
        }
        // Merge newly discovered columns back into the parent's cols.
        if (!allNewIds.empty()) {
            bumpInColsGen();
            for (int id : cols) markInCols(id);
            for (int id : allNewIds) {
                if (!isInCols(id) && id < (int)allColumns_.size()) {
                    cols.push_back(id);
                    markInCols(id);
                }
            }
        }
        return {bestV, bestA};
    }

    Solution reconstructIntegerSolution(const std::vector<int>& cols, const RmpResult& rmp) const {
        Solution sol;
        sol.feasible = true;
        sol.obj = rmp.obj;
        sol.alpha.assign(in_.m, std::vector<int>(in_.n, 0));
        sol.y.assign(in_.m, 0);
        sol.z.assign(in_.m, 0);
        sol.s.assign(in_.n, 0);
        sol.theta.assign(in_.numSucc, 0.0);
        sol.chosenPatterns.clear();

        for (int k = 0; k < (int)cols.size(); ++k) {
            if (rmp.lambdaValue[k] > 0.5) {
                const Pattern& p = allColumns_[cols[k]];
                sol.chosenPatterns.push_back(p);
                if (p.nonempty()) sol.y[p.v] = 1;
                for (int a : p.items) {
                    int j = activeOrig_[a];
                    sol.alpha[p.v][j] = 1;
                }
            }
        }
        for (int j = 0; j < in_.n; ++j) {
            if (!in_.active[j]) {
                sol.s[j] = 1; // exact match to uploaded Python MIP; set to 0 if you remove inactive fulfillment.
                continue;
            }
            int covered = 0;
            for (int v = 0; v < in_.m; ++v) covered += sol.alpha[v][j];
            sol.s[j] = covered ? 0 : 1;
        }
        assignOptimalPurchaseVector(sol);

        // Recompute theta exactly from the cuts.
        for (const Stage3Cut& cut : in_.cuts) {
            double rhs = cut.beta;
            for (int v = 0; v < in_.m; ++v) {
                rhs += cut.piY[v] * sol.y[v];
                for (int j = 0; j < in_.n; ++j) rhs += cut.piAlpha[v][j] * sol.alpha[v][j];
            }
            sol.theta[cut.succ] = std::max(sol.theta[cut.succ], rhs);
        }
        for (int h = 0; h < in_.numSucc; ++h) sol.theta[h] = std::max(std::max(0.0, params_.thetaLowerBound), sol.theta[h]);

        // Recompute objective to avoid relying on numerical LP values.
        double obj = 0.0;
        for (int h = 0; h < in_.numSucc; ++h) obj += sol.theta[h];
        for (int j = 0; j < in_.n; ++j) obj += in_.cOut[j] * sol.s[j];
        for (int v = 0; v < in_.m; ++v) obj -= in_.piZ[v] * sol.z[v];
        sol.obj = obj;
        return sol;
    }

    void applyBranchAlphaZero(BranchState& child, int v, int a) const {
        child.fixAlpha[v][a] = 0;
    }

    void applyBranchAlphaOne(BranchState& child, int v, int a) const {
        // alpha[v,a] = 1 implies no outsourcing and no other vehicle can take customer a.
        for (int u = 0; u < in_.m; ++u) child.fixAlpha[u][a] = 0;
        child.fixAlpha[v][a] = 1;
        child.fixY[v] = 1;
    }

    void applyBranchTogether(BranchState& child, int a, int b) const {
        child.togetherPairs.push_back(normPair(a, b));
    }

    void applyBranchSeparate(BranchState& child, int a, int b) const {
        child.separatePairs.push_back(normPair(a, b));
    }

    bool quickBranchFeasible(const BranchState& bs) const {
        // Check each active customer is not forced to more than one vehicle, and each
        // vehicle's forced set fits capacity.
        for (const auto& p1 : bs.togetherPairs) {
            const auto np1 = normPair(p1.first, p1.second);
            for (const auto& p2 : bs.separatePairs) {
                if (normPair(p2.first, p2.second) == np1) return false;
            }
        }

        for (int a = 0; a < activeN_; ++a) {
            int cnt = 0;
            for (int v = 0; v < in_.m; ++v) if (bs.fixAlpha[v][a] == 1) ++cnt;
            if (cnt > 1) return false;
        }
        for (int v = 0; v < in_.m; ++v) {
            double load = 0.0;
            int forced = 0;
            std::vector<int> forcedItems;
            for (int a = 0; a < activeN_; ++a) {
                if (bs.fixAlpha[v][a] == 1) {
                    load += in_.volume[activeOrig_[a]];
                    ++forced;
                    forcedItems.push_back(a);
                }
            }
            if (integerVolumeSumsExact_
                    ? load > in_.Qv[v]
                    : !exactActiveLoadAtMost(forcedItems, in_.Qv[v])) {
                return false;
            }
            if (bs.fixY[v] == 0 && forced > 0) return false;
        }

        // Fast consistency checks for Ryan-Foster constraints under current fixes.
        for (const auto& pr : bs.togetherPairs) {
            const int a = pr.first, b = pr.second;
            for (int v = 0; v < in_.m; ++v) {
                if ((bs.fixAlpha[v][a] == 1 && bs.fixAlpha[v][b] == 0) ||
                    (bs.fixAlpha[v][a] == 0 && bs.fixAlpha[v][b] == 1)) {
                    return false;
                }
            }
        }
        for (const auto& pr : bs.separatePairs) {
            const int a = pr.first, b = pr.second;
            for (int v = 0; v < in_.m; ++v) {
                if (bs.fixAlpha[v][a] == 1 && bs.fixAlpha[v][b] == 1) return false;
            }
        }
        return true;
    }

    // ==================================================================
    //  Cutting plane methods: separation, master insertion, CG-cut loop
    // ==================================================================

    void addSR3CutToMaster(int a, int b, int c) {
        GRBLinExpr lhs;
        // Use inverted index: union of columns touching a, b, or c
        std::vector<bool> seen(allColumns_.size(), false);
        auto check = [&](int cust) {
            for (int id : colsByCustomer_[cust]) {
                if (seen[id]) continue;
                seen[id] = true;
                const Pattern& p = allColumns_[id];
                int cnt = 0;
                if (containsActive(p, a)) ++cnt;
                if (containsActive(p, b)) ++cnt;
                if (containsActive(p, c)) ++cnt;
                if (cnt >= 2) lhs += colVars_[id];
            }
        };
        check(a); check(b); check(c);
        GRBConstr row = rmpModel_->addConstr(
            lhs <= 1.0, "sr3_" + std::to_string(sr3Cuts_.size()));
        sr3Rows_.push_back(row);
        sr3Cuts_.push_back({a, b, c});
        int ci = (int)sr3Cuts_.size() - 1;
        sr3ByCust_[a].emplace_back(ci, b, c);
        sr3ByCust_[b].emplace_back(ci, a, c);
        sr3ByCust_[c].emplace_back(ci, a, b);
        rmpModel_->update();
    }

    void addCoverCutToMaster(int vehicle, const std::vector<int>& cover,
                             const std::vector<int>& liftCoeffs) {
        GRBLinExpr lhs;
        for (int id : colsByVehicle_[vehicle]) {
            const Pattern& p = allColumns_[id];
            int coeff = 0;
            if (liftCoeffs.empty()) {
                for (int ca : cover) {
                    if (containsActive(p, ca)) ++coeff;
                }
            } else {
                for (int ia : p.items) {
                    if (ia < (int)liftCoeffs.size() && liftCoeffs[ia] > 0)
                        coeff += liftCoeffs[ia];
                }
            }
            if (coeff > 0) lhs += (double)coeff * colVars_[id];
        }
        GRBConstr row = rmpModel_->addConstr(
            lhs <= (double)(cover.size() - 1),
            "cover_" + std::to_string(coverCuts_.size()));
        coverRows_.push_back(row);
        coverCuts_.push_back({vehicle, cover, liftCoeffs});
        rmpModel_->update();
    }

    void addCliqueCutToMaster(const std::vector<int>& clique) {
        GRBLinExpr lhs;
        for (int ca : clique) lhs += oVars_[ca];
        GRBConstr row = rmpModel_->addConstr(
            lhs >= (double)((int)clique.size() - in_.m),
            "clique_" + std::to_string(cliqueCuts_.size()));
        cliqueRows_.push_back(row);
        cliqueCuts_.push_back({clique});
        rmpModel_->update();
    }

    int separateSR3Cuts(const RmpResult& rmp, const std::vector<int>& cols) {
        if ((int)sr3Cuts_.size() >= params_.maxSR3Cuts) return 0;

        // Build pairwise overlap matrix from LP solution.
        std::vector<std::vector<double>> xPair(activeN_,
            std::vector<double>(activeN_, 0.0));
        for (int k = 0; k < (int)cols.size(); ++k) {
            const double lam = rmp.lambdaValue[k];
            if (lam <= 1e-12) continue;
            const auto& itms = allColumns_[cols[k]].items;
            for (int i = 0; i < (int)itms.size(); ++i) {
                for (int j = i + 1; j < (int)itms.size(); ++j) {
                    xPair[itms[i]][itms[j]] += lam;
                    xPair[itms[j]][itms[i]] += lam;
                }
            }
        }

        auto alreadyHasSR3 = [&](int a, int b, int c) -> bool {
            for (const auto& sc : sr3Cuts_) {
                std::array<int,3> ex = {sc.a, sc.b, sc.c};
                std::sort(ex.begin(), ex.end());
                if (ex[0] == a && ex[1] == b && ex[2] == c) return true;
            }
            return false;
        };

        struct Violation { int a, b, c; double viol; };
        std::vector<Violation> violations;

        for (int a = 0; a < activeN_; ++a) {
            for (int b = a + 1; b < activeN_; ++b) {
                if (xPair[a][b] <= 0.01) continue;
                for (int c = b + 1; c < activeN_; ++c) {
                    double pairSum = xPair[a][b] + xPair[a][c] + xPair[b][c];
                    if (pairSum <= 1.0 + params_.cutViolationTol) continue;
                    if (alreadyHasSR3(a, b, c)) continue;

                    // Exact LHS: sum of lambda_k for columns hitting >= 2 of {a,b,c}
                    double lhs = 0.0;
                    for (int k = 0; k < (int)cols.size(); ++k) {
                        const double lam = rmp.lambdaValue[k];
                        if (lam <= 1e-12) continue;
                        const Pattern& p = allColumns_[cols[k]];
                        int cnt = 0;
                        if (containsActive(p, a)) ++cnt;
                        if (containsActive(p, b)) ++cnt;
                        if (containsActive(p, c)) ++cnt;
                        if (cnt >= 2) lhs += lam;
                    }
                    double viol = lhs - 1.0;
                    if (viol > params_.cutViolationTol) {
                        violations.push_back({a, b, c, viol});
                    }
                }
            }
        }

        std::sort(violations.begin(), violations.end(),
            [](const Violation& x, const Violation& y) { return x.viol > y.viol; });

        int added = 0;
        int maxThisRound = std::min(params_.maxSR3PerRound,
                                    params_.maxSR3Cuts - (int)sr3Cuts_.size());
        for (const auto& v : violations) {
            if (added >= maxThisRound) break;
            addSR3CutToMaster(v.a, v.b, v.c);
            ++added;
        }
        return added;
    }

    int separateCoverCuts(const RmpResult& rmp, const std::vector<int>& cols) {
        if ((int)coverCuts_.size() >= params_.maxCoverCuts) return 0;
        int added = 0;
        int maxThisRound = std::min(params_.maxCoverPerRound,
                                    params_.maxCoverCuts - (int)coverCuts_.size());

        for (int v = 0; v < in_.m; ++v) {
            if (added >= maxThisRound) break;

            struct CustVal { int a; double val; double vol; };
            std::vector<CustVal> cands;
            for (int a = 0; a < activeN_; ++a) {
                double val = rmp.alphaValue[v][a];
                if (val > 1e-8)
                    cands.push_back({a, val, in_.volume[activeOrig_[a]]});
            }
            std::sort(cands.begin(), cands.end(),
                [](const CustVal& x, const CustVal& y) { return x.val > y.val; });

            std::vector<int> cover;
            double sumVol = 0.0;
            for (const auto& cv : cands) {
                cover.push_back(cv.a);
                sumVol += cv.vol;
                if (sumVol > in_.Qv[v] + EPS) break;
            }
            if (sumVol <= in_.Qv[v] + EPS) continue;
            if (cover.size() <= 1) continue;

            std::vector<int> sortedCover = cover;
            std::sort(sortedCover.begin(), sortedCover.end());
            bool dup = false;
            for (const auto& cc : coverCuts_) {
                if (cc.vehicle != v) continue;
                if (cc.cover == sortedCover) { dup = true; break; }
            }
            if (dup) continue;

            // Use basic (non-lifted) cover inequality only.
            // Independent lifting of non-cover items is invalid when multiple
            // lifted items are assigned simultaneously (they interact through
            // shared capacity, making the lifted inequality too tight).
            std::vector<int> liftCoeffs;  // empty = no lifting

            // Compute LHS (basic cover inequality: count cover members)
            double lhs = 0.0;
            for (int k = 0; k < (int)cols.size(); ++k) {
                const double lam = rmp.lambdaValue[k];
                if (lam <= 1e-12) continue;
                const Pattern& p = allColumns_[cols[k]];
                if (p.v != v) continue;
                int coeff = 0;
                for (int ca : sortedCover) {
                    if (containsActive(p, ca)) ++coeff;
                }
                lhs += coeff * lam;
            }
            double rhs = (double)((int)cover.size() - 1);
            if (lhs > rhs + params_.cutViolationTol) {
                addCoverCutToMaster(v, sortedCover, liftCoeffs);
                ++added;
            }
        }
        return added;
    }

    int separateCliqueCuts(const RmpResult& rmp) {
        if ((int)cliqueCuts_.size() >= params_.maxCliqueCuts) return 0;

        double Qmax = *std::max_element(in_.Qv.begin(), in_.Qv.end());
        double halfQ = Qmax / 2.0;

        std::vector<int> clique;
        for (int a = 0; a < activeN_; ++a) {
            if (in_.volume[activeOrig_[a]] > halfQ + EPS)
                clique.push_back(a);
        }
        if ((int)clique.size() <= in_.m) return 0;

        double lhsVal = 0.0;
        for (int ca : clique) lhsVal += rmp.oValue[ca];
        double rhsVal = (double)((int)clique.size() - in_.m);
        if (lhsVal >= rhsVal - params_.cutViolationTol) return 0;

        std::sort(clique.begin(), clique.end());
        for (const auto& cc : cliqueCuts_) {
            if (cc.clique == clique) return 0;
        }
        addCliqueCutToMaster(clique);
        return 1;
    }

    int separateAndAddCuts(const RmpResult& rmp, const std::vector<int>& cols, int depth) {
        if (depth > params_.maxCutDepth) return 0;
        // Diagnostic env toggles (default: all cuts on). Used to bisect a
        // platform-dependent LB-violation: VRP_BPC_NO_CUTS=1 disables all,
        // VRP_BPC_NO_{SR3,COVER,CLIQUE}=1 disable a single family.
        static const bool noCuts   = (getenv("VRP_BPC_NO_CUTS")   != nullptr);
        static const bool noSR3    = (getenv("VRP_BPC_NO_SR3")    != nullptr);
        static const bool noCover  = (getenv("VRP_BPC_NO_COVER")  != nullptr);
        static const bool noClique = (getenv("VRP_BPC_NO_CLIQUE") != nullptr);
        if (noCuts) return 0;
        int total = 0;
        if (params_.useSR3Cuts && !noSR3)
            total += separateSR3Cuts(rmp, cols);
        if (params_.useCoverCuts && !noCover)
            total += separateCoverCuts(rmp, cols);
        if (params_.useCliqueCuts && !noClique)
            total += separateCliqueCuts(rmp);
        return total;
    }

    void ageCuts(const RmpResult& rmp) {
        if (!params_.useCutAging) return;
        for (int ci = 0; ci < (int)sr3Cuts_.size(); ++ci) {
            if (!sr3Cuts_[ci].active) continue;
            double dualAbs = std::fabs(rmp.dualSR3[ci]);
            sr3Cuts_[ci].lastDualAbs = dualAbs;
            if (dualAbs < params_.cutDualTol) {
                ++sr3Cuts_[ci].age;
                if (sr3Cuts_[ci].age >= params_.cutAgingRounds) {
                    sr3Cuts_[ci].active = false;
                    sr3Rows_[ci].set(GRB_CharAttr_Sense, '<');
                    sr3Rows_[ci].set(GRB_DoubleAttr_RHS, GRB_INFINITY);
                }
            } else {
                sr3Cuts_[ci].age = 0;
            }
        }
        for (int ci = 0; ci < (int)coverCuts_.size(); ++ci) {
            if (!coverCuts_[ci].active) continue;
            double dualAbs = std::fabs(rmp.dualCover[ci]);
            coverCuts_[ci].lastDualAbs = dualAbs;
            if (dualAbs < params_.cutDualTol) {
                ++coverCuts_[ci].age;
                if (coverCuts_[ci].age >= params_.cutAgingRounds) {
                    coverCuts_[ci].active = false;
                    coverRows_[ci].set(GRB_CharAttr_Sense, '<');
                    coverRows_[ci].set(GRB_DoubleAttr_RHS, GRB_INFINITY);
                }
            } else {
                coverCuts_[ci].age = 0;
            }
        }
        rmpModel_->update();
    }

        bool cutAndPrice(const BranchState& bs, std::vector<int>& cols,
                     RmpResult& finalRmp, int depth) {
        for (int round = 0; round <= params_.maxCuttingRounds; ++round) {
            bool cgOk = columnGeneration(bs, cols, finalRmp, depth);
            if (backward_gap_stop_) return false;
            if (timed_out_) return false;
            if (!cgOk || !finalRmp.feasible) return false;
            if (params_.forwardUbMode && !last_cg_certified_
                    && !finalRmp.dualBoundCertified)
                return true;

            // Preserve every fully priced root relaxation immediately.  If a
            // later cut round times out, the strongest already completed root
            // bound is still a valid global LB and must not be discarded.
            if (depth == 0 && (last_cg_certified_ || params_.forwardUbMode)
                    && finalRmp.dualBoundCertified) {
                if (last_cg_certified_) ++root_pricing_passes_completed_;
                root_cut_rounds_completed_ = round;
                if (!root_lb_certified_
                        || finalRmp.certifiedDualBound > global_lb_) {
                    global_lb_ = finalRmp.certifiedDualBound;
                    root_lp_after_cuts_ = finalRmp.certifiedDualBound;
                    root_columns_ = (int)cols.size();
                }
                root_lb_certified_ = true;
            }

            if (depth == 0 && round == 0) {
                root_lp_before_cuts_ = finalRmp.obj;
            }

            // A fully priced root may already supply an integer policy.
            // Audit/reprice it before doing another separation/pricing round
            // when the caller requested only a backward query interval.
            if (!params_.forwardUbMode
                    && (params_.backwardGapAbs > 0.0 || params_.backwardGapRel > 0.0)
                    && finalRmp.feasible && isIntegerRmp(finalRmp)) {
                Solution candidate = reconstructIntegerSolution(cols, finalRmp);
                if (fullFeasibilityCheck(candidate, bs)) {
                    candidate.ub_certified = true;
                    if (!best_.feasible || candidate.obj < best_.obj)
                        best_ = std::move(candidate);
                }
            }
            if (backwardTargetSatisfied()) return true;

            // Column pool culling after CG converges
            if (round == 0) {
                cullHighRcColumns(cols, finalRmp);
            }

            // Cut aging: deactivate inactive cuts
            ageCuts(finalRmp);

            if (round == params_.maxCuttingRounds) break;
            if (depth > params_.maxCutDepth) break;

            const auto t_cut0 = std::chrono::steady_clock::now();
            int cutsAdded = separateAndAddCuts(finalRmp, cols, depth);
            cut_separation_time_s_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_cut0).count();
            if (params_.verbose && cutsAdded > 0) {
                std::cerr << "  Cut round " << round
                          << ": added " << cutsAdded << " cuts"
                          << " (SR3=" << sr3Cuts_.size()
                          << " Cover=" << coverCuts_.size()
                          << " Clique=" << cliqueCuts_.size() << ")\n";
            }
            if (cutsAdded == 0) break;
        }
        return true;
    }

    bool fullFeasibilityCheck(Solution& sol, const BranchState& bs) const {
        if (!sol.feasible) return false;
        // 1. Fulfillment: each customer assigned or outsourced exactly once
        for (int j = 0; j < in_.n; ++j) {
            int count = sol.s[j];
            for (int v = 0; v < in_.m; ++v) count += sol.alpha[v][j];
            if (count != 1) return false;
        }
        // 2. No multi-vehicle assignment
        for (int j = 0; j < in_.n; ++j) {
            int cnt = 0;
            for (int v = 0; v < in_.m; ++v) cnt += sol.alpha[v][j];
            if (cnt > 1) return false;
        }
        // 3. Capacity
        for (int v = 0; v < in_.m; ++v) {
            std::vector<double> assignedVolumes;
            for (int j = 0; j < in_.n; ++j) {
                if (sol.alpha[v][j]) assignedVolumes.push_back(in_.volume[j]);
            }
            if (pricing_detail::compareNonnegativeBinary64Sum(
                    assignedVolumes, in_.Qv[v]) > 0) return false;
        }
        // 4. Activation logic
        for (int v = 0; v < in_.m; ++v) {
            if (sol.y[v] > sol.z[v]) return false;
            int assigned = 0;
            for (int j = 0; j < in_.n; ++j)
                if (sol.alpha[v][j]) ++assigned;
            if (assigned > 0 && !sol.y[v]) return false;
            if (sol.y[v] && assigned == 0) return false;
            for (int j = 0; j < in_.n; ++j)
                if (sol.alpha[v][j] && !in_.active[j]) return false;
        }
        if (params_.usePurchaseOrder) {
            for (const auto& vehicles : purchaseGroups_) {
                for (int rank = 1;
                        rank < static_cast<int>(vehicles.size()); ++rank) {
                    if (sol.z[vehicles[rank - 1]] < sol.z[vehicles[rank]])
                        return false;
                }
            }
        }
        // 5. Branching restrictions
        for (int v = 0; v < in_.m; ++v) {
            for (int a = 0; a < activeN_; ++a) {
                int j = activeOrig_[a];
                if (bs.fixAlpha[v][a] == 0 && sol.alpha[v][j]) return false;
                if (bs.fixAlpha[v][a] == 1 && !sol.alpha[v][j]) return false;
            }
            if (bs.fixY[v] == 0 && sol.y[v]) return false;
            if (bs.fixY[v] == 1 && !sol.y[v]) return false;
        }
        for (const auto& pr : bs.togetherPairs) {
            for (int v = 0; v < in_.m; ++v) {
                bool ha = sol.alpha[v][activeOrig_[pr.first]] != 0;
                bool hb = sol.alpha[v][activeOrig_[pr.second]] != 0;
                if (ha != hb) return false;
            }
        }
        for (const auto& pr : bs.separatePairs) {
            for (int v = 0; v < in_.m; ++v) {
                if (sol.alpha[v][activeOrig_[pr.first]] && sol.alpha[v][activeOrig_[pr.second]])
                    return false;
            }
        }
        // 6-8. Recompute theta from ALL Benders cuts (including dormant).
        // This guarantees theta satisfies every cut; no separate violation check needed.
        std::vector<double> thetaRecomputed(
            in_.numSucc,
            pricing_detail::roundUp(
                std::max(0.0, params_.thetaLowerBound)));
        for (const Stage3Cut& cut : in_.cuts) {
            double rhs = pricing_detail::roundUp(cut.beta);
            for (int v = 0; v < in_.m; ++v) {
                if (sol.y[v])
                    rhs = pricing_detail::addUp(rhs, cut.piY[v]);
                for (int j = 0; j < in_.n; ++j) {
                    if (sol.alpha[v][j])
                        rhs = pricing_detail::addUp(
                            rhs, cut.piAlpha[v][j]);
                }
            }
            thetaRecomputed[cut.succ] = std::max(thetaRecomputed[cut.succ], rhs);
        }
        sol.theta = thetaRecomputed;
        // 9. Recompute a directed-up objective.  `sol.obj` is therefore a
        // mathematical UB, not an ordinary floating sum that could round below
        // the true cost and spuriously meet a directed-down LB.
        double obj = 0.0;
        for (int h = 0; h < in_.numSucc; ++h)
            obj = pricing_detail::addUp(obj, sol.theta[h]);
        for (int j = 0; j < in_.n; ++j) {
            if (sol.s[j])
                obj = pricing_detail::addUp(obj, in_.cOut[j]);
        }
        for (int v = 0; v < in_.m; ++v) {
            if (sol.z[v])
                obj = pricing_detail::addUp(obj, -in_.piZ[v]);
        }
        sol.obj = obj;
        return true;
    }

    // Restricted integer master heuristic: solve small MIP on current column pool
    Solution restrictedIntegerMaster(const std::vector<int>& cols,
                                      const BranchState& bs) {
        const auto t0 = std::chrono::steady_clock::now();
        Solution sol;
        sol.feasible = false;
        auto finished = [&]() {
            primal_heuristic_time_s_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            return sol;
        };
        if (checkTimeout()) return finished();
        try {
            GRBModel mip(env_);
            mip.set(GRB_IntParam_OutputFlag, 0);
            mip.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);
            mip.set(GRB_DoubleParam_TimeLimit, params_.restrictedMipTimeLimit);

            std::vector<GRBVar> lamVars(cols.size());
            for (int k = 0; k < (int)cols.size(); ++k) {
                if ((k & 255) == 0 && checkTimeout()) return finished();
                const Pattern& p = allColumns_[cols[k]];
                double obj = deltaY_[p.v] * (p.nonempty() ? 1.0 : 0.0);
                lamVars[k] = mip.addVar(0, 1, obj, GRB_BINARY,
                    "lam_" + std::to_string(k));
            }
            std::vector<GRBVar> mipO(activeN_);
            for (int a = 0; a < activeN_; ++a) {
                int j = activeOrig_[a];
                mipO[a] = mip.addVar(0, 1, in_.cOut[j], GRB_BINARY,
                    "o_" + std::to_string(a));
            }
            std::vector<GRBVar> mipTheta(in_.numSucc);
            for (int h = 0; h < in_.numSucc; ++h) {
                mipTheta[h] = mip.addVar(
                    std::max(0.0, params_.thetaLowerBound),
                    GRB_INFINITY, 1.0, GRB_CONTINUOUS,
                    "theta_" + std::to_string(h));
            }
            mip.update();

            // Vehicle convexity: sum of lambda for each vehicle == 1
            for (int v = 0; v < in_.m; ++v) {
                GRBLinExpr expr;
                for (int k = 0; k < (int)cols.size(); ++k)
                    if (allColumns_[cols[k]].v == v) expr += lamVars[k];
                mip.addConstr(expr == 1.0);
            }
            // Customer coverage
            for (int a = 0; a < activeN_; ++a) {
                GRBLinExpr expr = mipO[a];
                for (int k = 0; k < (int)cols.size(); ++k) {
                    if (containsActive(allColumns_[cols[k]], a))
                        expr += lamVars[k];
                }
                mip.addConstr(expr == 1.0);
            }
            // Benders cuts
            for (int r = 0; r < (int)in_.cuts.size(); ++r) {
                if ((r & 31) == 0 && checkTimeout()) return finished();
                const Stage3Cut& cut = in_.cuts[r];
                GRBLinExpr expr = mipTheta[cut.succ];
                for (int k = 0; k < (int)cols.size(); ++k) {
                    double coeff = cutCoeff(cut, allColumns_[cols[k]]);
                    if (coeff != 0.0)
                        expr -= coeff * lamVars[k];
                }
                mip.addConstr(expr >= cut.beta);
            }
            if (params_.forwardUbMode) {
                Solution start = best_;
                if (start.feasible && fullFeasibilityCheck(start, bs)) {
                    std::vector<int> startIndex(in_.m, -1);
                    for (int k = 0; k < (int)cols.size(); ++k) {
                        const Pattern& p = allColumns_[cols[k]];
                        bool equal = true;
                        for (int a = 0; a < activeN_; ++a) {
                            if (containsActive(p, a)
                                    != (start.alpha[p.v][activeOrig_[a]] != 0)) {
                                equal = false;
                                break;
                            }
                        }
                        if (equal) startIndex[p.v] = k;
                    }
                    if (std::find(startIndex.begin(), startIndex.end(), -1)
                            == startIndex.end()) {
                        for (int k = 0; k < (int)cols.size(); ++k)
                            lamVars[k].set(GRB_DoubleAttr_Start,
                                startIndex[allColumns_[cols[k]].v] == k ? 1.0 : 0.0);
                        for (int a = 0; a < activeN_; ++a)
                            mipO[a].set(GRB_DoubleAttr_Start, start.s[activeOrig_[a]]);
                        for (int h = 0; h < in_.numSucc; ++h)
                            mipTheta[h].set(GRB_DoubleAttr_Start, start.theta[h]);
                    }
                }
            }
            if (checkTimeout()) return finished();
            mip.set(GRB_DoubleParam_TimeLimit,
                std::min(params_.restrictedMipTimeLimit, remainingTime()));
            mip.optimize();
            int status = mip.get(GRB_IntAttr_Status);
            if (params_.verbose) {
                std::cerr << "  RestrictedMIP cols=" << cols.size()
                          << " status=" << status
                          << " solutions=" << mip.get(GRB_IntAttr_SolCount);
                if (mip.get(GRB_IntAttr_SolCount) > 0)
                    std::cerr << " objective=" << mip.get(GRB_DoubleAttr_ObjVal)
                              + constantZ_ + inactiveOutCost_;
                std::cerr << "\n";
            }
            if (status == GRB_OPTIMAL || status == GRB_TIME_LIMIT) {
                if (mip.get(GRB_IntAttr_SolCount) > 0) {
                    sol = reconstructFromRestrictedMip(
                        cols, lamVars, mipO, mipTheta, mip);
                }
            }
        } catch (const GRBException&) {
        }
        return finished();
    }

    Solution reconstructFromRestrictedMip(
        const std::vector<int>& cols,
        const std::vector<GRBVar>& lamVars,
        const std::vector<GRBVar>& mipO,
        const std::vector<GRBVar>& mipTheta,
        GRBModel& mip) const {
        Solution sol;
        sol.feasible = true;
        sol.alpha.assign(in_.m, std::vector<int>(in_.n, 0));
        sol.y.assign(in_.m, 0);
        sol.z.assign(in_.m, 0);
        sol.s.assign(in_.n, 0);
        sol.theta.assign(in_.numSucc, 0.0);

        for (int k = 0; k < (int)cols.size(); ++k) {
            if (lamVars[k].get(GRB_DoubleAttr_X) > 0.5) {
                const Pattern& p = allColumns_[cols[k]];
                if (p.nonempty()) sol.y[p.v] = 1;
                for (int a : p.items) {
                    int j = activeOrig_[a];
                    sol.alpha[p.v][j] = 1;
                }
            }
        }
        for (int j = 0; j < in_.n; ++j) {
            if (!in_.active[j]) { sol.s[j] = 1; continue; }
            int covered = 0;
            for (int v = 0; v < in_.m; ++v) covered += sol.alpha[v][j];
            sol.s[j] = covered ? 0 : 1;
        }
        assignOptimalPurchaseVector(sol);
        // Recompute theta from all cuts
        for (const Stage3Cut& cut : in_.cuts) {
            double rhs = cut.beta;
            for (int v = 0; v < in_.m; ++v) {
                rhs += cut.piY[v] * sol.y[v];
                for (int j = 0; j < in_.n; ++j)
                    rhs += cut.piAlpha[v][j] * sol.alpha[v][j];
            }
            sol.theta[cut.succ] = std::max(sol.theta[cut.succ], rhs);
        }
        for (int h = 0; h < in_.numSucc; ++h)
            sol.theta[h] = std::max(sol.theta[h],
                std::max(0.0, params_.thetaLowerBound));
        double obj = 0.0;
        for (int h = 0; h < in_.numSucc; ++h) obj += sol.theta[h];
        for (int j = 0; j < in_.n; ++j) obj += in_.cOut[j] * sol.s[j];
        for (int v = 0; v < in_.m; ++v) obj -= in_.piZ[v] * sol.z[v];
        sol.obj = obj;
        return sol;
    }

    void branchAndPrice(const BranchState& bs, std::vector<int> cols,
                        int depth, double inherited_lb = -INF) {
        if (backwardTargetSatisfied()) return;
        if (checkTimeout()) {
            ++pricing_uncertified_nodes_;
            return;
        }
        if (++nodesProcessed_ > params_.maxNodes) {
            timed_out_ = true;
            ++pricing_uncertified_nodes_;
            return;
        }
        if (depth > params_.maxDepth) {
            timed_out_ = true;
            ++pricing_uncertified_nodes_;
            return;
        }
        if (!quickBranchFeasible(bs)) return;

        // Forward UB mode: global "good enough" early stop. global_lb_ is a
        // VALID global lower bound (the certified root LP bound). Once the
        // incumbent is within forwardGap of it, the remaining tree cannot
        // improve the UB by more than the tolerated gap, so stop exploring.
        if (params_.forwardUbMode && best_.feasible
                && params_.forwardGap > 0.0 && global_lb_ > -INF + 1.0) {
            const double slack = params_.forwardGap * std::max(1.0, std::fabs(global_lb_));
            if (best_.obj - global_lb_ <= slack) {
                used_gap_prune_ = true;
                proof_relaxed_ = true;
                ++tolerance_bound_prunes_;
                return;
            }
        }

        if (params_.verbose) {
            std::cerr << "Node " << nodesProcessed_ << " depth=" << depth
                      << " incumbent=" << best_.obj
                      << " inherited_lb=" << inherited_lb << "\n";
        }

        // Prune by inherited LB (always valid)
        if (shouldPruneByBound(inherited_lb)) return;

        RmpResult rmp;
        const bool cgOk = cutAndPrice(bs, cols, rmp, depth);
        if (backwardTargetSatisfied()) return;
        if (timed_out_) {
            ++pricing_uncertified_nodes_;
            ++used_inherited_lb_nodes_;
            return;
        }
        if (!cgOk || !rmp.feasible) {
            ++pricing_uncertified_nodes_;
            return;
        }

        // In forward UB mode, depth>0 nodes may have converged via heuristic-only
        // pricing (last_cg_certified_ == false). Then rmp.obj is a subset-of-
        // columns value (>= true LP optimum), i.e. NOT a valid lower bound, so we
        // fall back to the parent's valid inherited_lb for every bound decision.
        const bool node_certified = (last_cg_certified_ || params_.forwardUbMode)
            && rmp.dualBoundCertified
            && std::isfinite(rmp.certifiedDualBound);
        const double node_lb = node_certified
            ? rmp.certifiedDualBound : inherited_lb;
        if (node_certified && last_cg_certified_) {
            ++pricing_certified_nodes_;
        } else {
            ++pricing_uncertified_nodes_;
            ++used_inherited_lb_nodes_;
        }

        if (depth == 0 && node_certified) {
            global_lb_ = std::max(global_lb_, rmp.certifiedDualBound);
            root_lb_certified_ = true;
            root_lp_after_cuts_ = global_lb_;
            root_columns_ = (int)cols.size();
        }

        // Backward root-bound-only mode deliberately stops here.  cutAndPrice
        // has already completed exact raw-dual pricing for the root after every
        // accepted cut round, so global_lb_ is a certified relaxation bound.
        // Do not enter diving, strong branching, or the child tree: those can
        // improve an incumbent/proof but cannot improve the returned global LB
        // in this implementation.  Keep this distinct from timed_out_.
        if (depth == 0 && params_.rootBoundOnly) {
            // Defensive invariant: root-only is useful only after exact root
            // pricing.  Never label a heuristic/subset-column root return as an
            // intentional certified stop, even if a future solve-mode change
            // makes such a state reachable.
            if (!node_certified) {
                timed_out_ = true;
                best_.abort_reason = "root_bound_only_without_certified_pricing";
                return;
            }
            intentional_root_stop_ = true;

            // The sole exception is an integral, fully checked root solution.
            // In that case the root relaxation itself closes the entire tree.
            // Merely observing incumbent == LB is intentionally insufficient:
            // numerical coincidence must never turn a root-only run into an
            // optimality certificate.
            if (node_certified && isIntegerRmp(rmp)) {
                Solution rootSol = reconstructIntegerSolution(cols, rmp);
                if (fullFeasibilityCheck(rootSol, bs)) {
                    rootSol.ub_certified = true;
                    const double certifiedGap =
                        rootSol.obj - rmp.certifiedDualBound;
                    if (certifiedGap <= 0.0) {
                        root_integral_proof_ = true;
                        // Use the same reconstructed root solution that closed
                        // the relaxation, even if a numerically equal greedy
                        // incumbent was already present.  This keeps the exact
                        // certificate and returned incumbent tied together.
                        best_ = std::move(rootSol);
                    } else {
                        const double formerTolerance = 1e-6
                            * (1.0 + std::fabs(rmp.certifiedDualBound));
                        if (certifiedGap <= formerTolerance) {
                            proof_relaxed_ = true;
                            ++tolerance_integral_closures_;
                        }
                        if (!best_.feasible || rootSol.obj < best_.obj)
                            best_ = std::move(rootSol);
                    }
                }
            }
            return;
        }

        if (shouldPruneByBound(node_lb)) return;

        if (isIntegerRmp(rmp)) {
            Solution sol = reconstructIntegerSolution(cols, rmp);
            bool integerFeasible = false;
            double integerObjective = INF;
            if (fullFeasibilityCheck(sol, bs)) {
                integerFeasible = true;
                integerObjective = sol.obj;
                sol.ub_certified = true;
                if (!best_.feasible || sol.obj < best_.obj) {
                    best_ = std::move(sol);
                    if (params_.verbose) std::cerr << "  New incumbent: " << best_.obj << "\n";
                }
            }
            if (backwardTargetSatisfied()) return;
            // intTol only says the LP is near an integer point.  It may close
            // this node for UB search, but it is a strict proof only when a
            // fully checked integer objective is no larger than the certified
            // node lower bound.  Otherwise retain the safe root LB and prevent
            // solve() from upgrading it to the incumbent objective.
            if (!(integerFeasible && node_certified
                    && integerObjective <= node_lb)) {
                proof_relaxed_ = true;
                ++tolerance_integral_closures_;
            }
            return;
        }

        // Diving heuristic
        if (params_.useDiving && depth <= params_.divingMaxDepth) {
            Solution dive = divingHeuristic(cols, rmp);
            if (dive.feasible && fullFeasibilityCheck(dive, bs)) {
                dive.ub_certified = true;
                if (!best_.feasible || dive.obj < best_.obj) {
                    best_ = std::move(dive);
                    if (params_.verbose) std::cerr << "  Diving incumbent: " << best_.obj << "\n";
                }
            }
            if (backwardTargetSatisfied()) return;
            if (shouldPruneByBound(node_lb)) return;
        }

        // Restricted integer master heuristic
        if (params_.useRestrictedMip
            && nodesProcessed_ % params_.restrictedMipFreq == 0) {
            Solution rmipSol = restrictedIntegerMaster(cols, bs);
            if (rmipSol.feasible && fullFeasibilityCheck(rmipSol, bs)) {
                rmipSol.ub_certified = true;
                if (!best_.feasible || rmipSol.obj < best_.obj) {
                    best_ = std::move(rmipSol);
                    if (params_.verbose) std::cerr << "  RestrictedMIP incumbent: " << best_.obj << "\n";
                }
            }
            if (backwardTargetSatisfied()) return;
            if (shouldPruneByBound(node_lb)) return;
        }

        // Branching. On fleets with many identical vehicles, alpha[v]
        // branching is symmetric (vehicle-permutation blow-up). Ryan-Foster
        // pair branching is vehicle-agnostic / symmetry-free, so in forward
        // (preferRyanFoster) mode we try it FIRST and only fall back to
        // alpha[v]. In default/backward mode the original order is kept.
        int branchV = -1, branchA = -1;

        if (params_.preferRyanFoster && params_.useRyanFoster) {
            auto [rfA, rfB] = chooseRyanFosterPair(rmp, cols, bs);
            if (rfA >= 0) {
                BranchState together = bs;
                applyBranchTogether(together, rfA, rfB);
                std::vector<int> colsTogether = cols;
                branchAndPrice(together, colsTogether, depth + 1, node_lb);
                if (timed_out_) return;

                BranchState separate = bs;
                applyBranchSeparate(separate, rfA, rfB);
                std::vector<int> colsSeparate = cols;
                branchAndPrice(separate, colsSeparate, depth + 1, node_lb);
                return;
            }
            // Forward UB mode: no Ryan-Foster pair available. Do NOT fall back
            // to symmetric alpha[v] branching once we already have a feasible
            // incumbent -- that reintroduces the vehicle-permutation blow-up we
            // are trying to avoid. The incumbent is valid; just close the node.
            // (When no incumbent exists yet we still fall through to alpha
            // branching below so we are guaranteed to find a feasible solution.)
            if (params_.forwardUbMode && best_.feasible) {
                // This is an intentional UB-only early close, not an
                // optimality proof.  Keep the feasible incumbent, but force
                // solve() to return the certified root LB rather than relabel
                // the incumbent objective as an exact lower bound.
                used_gap_prune_ = true;
                proof_relaxed_ = true;
                ++tolerance_bound_prunes_;
                return;
            }
        }

        // alpha[v][a] branching (primary in default/backward, fallback in forward)
        std::tie(branchV, branchA) = chooseAlphaBranchStrong(rmp, bs, cols, depth);
        if (branchV < 0 && !timed_out_) {
            std::tie(branchV, branchA) = chooseAlphaBranch(rmp, bs);
        }

        if (branchV < 0 && params_.useRyanFoster && !params_.preferRyanFoster) {
            auto [rfA, rfB] = chooseRyanFosterPair(rmp, cols, bs);
            if (rfA >= 0) {
                BranchState together = bs;
                applyBranchTogether(together, rfA, rfB);
                std::vector<int> colsTogether = cols;
                branchAndPrice(together, colsTogether, depth + 1, node_lb);
                if (timed_out_) return;

                BranchState separate = bs;
                applyBranchSeparate(separate, rfA, rfB);
                std::vector<int> colsSeparate = cols;
                branchAndPrice(separate, colsSeparate, depth + 1, node_lb);
                return;
            }
        }

        if (branchV < 0) {
            // The RMP is fractional (handled above by isIntegerRmp), but the
            // current branching rules could not identify a child split.  This
            // is not a proof of optimality; mark the solve as incomplete so
            // Python will not use the incumbent objective as a certified LB.
            if (params_.verbose) std::cerr << "  No branch candidate; solve incomplete.\n";
            timed_out_ = true;
            ++pricing_uncertified_nodes_;
            return;
        }

        BranchState one = bs;
        applyBranchAlphaOne(one, branchV, branchA);
        std::vector<int> colsOne = cols;
        branchAndPrice(one, colsOne, depth + 1, node_lb);
        if (timed_out_) return;

        BranchState zero = bs;
        applyBranchAlphaZero(zero, branchV, branchA);
        std::vector<int> colsZero = cols;
        branchAndPrice(zero, colsZero, depth + 1, node_lb);
    }

    Solution greedyIncumbent() const {
        // Simple assignment heuristic: sort active customers by cOut/volume and put them
        // into the vehicle with largest residual capacity that can fit. Then evaluate cuts.
        Solution sol;
        sol.feasible = true;
        sol.alpha.assign(in_.m, std::vector<int>(in_.n, 0));
        sol.y.assign(in_.m, 0);
        sol.z.assign(in_.m, 0);
        sol.s.assign(in_.n, 1);
        sol.theta.assign(in_.numSucc, 0.0);

        std::vector<double> rem = in_.Qv;
        std::vector<int> order(activeN_);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            int ja = activeOrig_[a], jb = activeOrig_[b];
            double ra = in_.cOut[ja] / std::max(in_.volume[ja], 1e-12);
            double rb = in_.cOut[jb] / std::max(in_.volume[jb], 1e-12);
            return ra > rb;
        });
        for (int a : order) {
            int j = activeOrig_[a];
            int bestV = -1;
            double bestRem = -1.0;
            for (int v = 0; v < in_.m; ++v) {
                if (!compatible_[v][a] || rem[v] + EPS < in_.volume[j]) continue;
                if (rem[v] > bestRem) {
                    bestRem = rem[v];
                    bestV = v;
                }
            }
            if (bestV >= 0) {
                sol.alpha[bestV][j] = 1;
                sol.y[bestV] = 1;
                sol.s[j] = 0;
                rem[bestV] -= in_.volume[j];
            }
        }
        for (int j = 0; j < in_.n; ++j) {
            if (!in_.active[j]) sol.s[j] = 1;
        }
        assignOptimalPurchaseVector(sol);
        for (const Stage3Cut& cut : in_.cuts) {
            double rhs = cut.beta;
            for (int v = 0; v < in_.m; ++v) {
                rhs += cut.piY[v] * sol.y[v];
                for (int j = 0; j < in_.n; ++j) rhs += cut.piAlpha[v][j] * sol.alpha[v][j];
            }
            sol.theta[cut.succ] = std::max(sol.theta[cut.succ], rhs);
        }
        for (double& th : sol.theta) th = std::max(std::max(0.0, params_.thetaLowerBound), th);

        double obj = 0.0;
        for (double th : sol.theta) obj += th;
        for (int j = 0; j < in_.n; ++j) obj += in_.cOut[j] * sol.s[j];
        for (int v = 0; v < in_.m; ++v) obj -= in_.piZ[v] * sol.z[v];
        sol.obj = obj;
        return sol;
    }
};

Solution solveGurobiMip(const Stage2Input& in, bool verbose = false) {
    Solution sol;
    try {
        GRBEnv env(true);
        env.set(GRB_IntParam_OutputFlag, verbose ? 1 : 0);
        env.start();
        GRBModel model(env);

        const int n = in.n, m = in.m;

        std::vector<std::vector<GRBVar>> alpha(n, std::vector<GRBVar>(m));
        std::vector<GRBVar> s(n), y(m), z(m);
        std::vector<GRBVar> theta(in.numSucc);

        for (int j = 0; j < n; ++j)
            for (int v = 0; v < m; ++v)
                alpha[j][v] = model.addVar(0, 1, 0, GRB_BINARY, "alpha_" + std::to_string(j) + "_" + std::to_string(v));
        for (int j = 0; j < n; ++j)
            s[j] = model.addVar(0, 1, in.cOut[j], GRB_BINARY, "s_" + std::to_string(j));
        for (int v = 0; v < m; ++v)
            y[v] = model.addVar(0, 1, 0, GRB_BINARY, "y_" + std::to_string(v));
        for (int v = 0; v < m; ++v)
            z[v] = model.addVar(0, 1, -in.piZ[v], GRB_BINARY, "z_" + std::to_string(v));
        for (int h = 0; h < in.numSucc; ++h)
            theta[h] = model.addVar(0, GRB_INFINITY, 1.0, GRB_CONTINUOUS, "theta_" + std::to_string(h));
        model.update();

        for (int j = 0; j < n; ++j) {
            GRBLinExpr expr = s[j];
            for (int v = 0; v < m; ++v) expr += alpha[j][v];
            model.addConstr(expr == 1, "fulfillment_" + std::to_string(j));
        }

        for (int v = 0; v < m; ++v)
            model.addConstr(y[v] <= z[v], "activation_" + std::to_string(v));

        for (int j = 0; j < n; ++j)
            for (int v = 0; v < m; ++v)
                model.addConstr(alpha[j][v] <= y[v], "assign_avail_" + std::to_string(j) + "_" + std::to_string(v));

        for (int v = 0; v < m; ++v) {
            GRBLinExpr expr = 0;
            for (int j = 0; j < n; ++j) expr += in.volume[j] * alpha[j][v];
            model.addConstr(expr <= in.Qv[v] * y[v], "capacity_" + std::to_string(v));
        }

        for (int v = 0; v < m; ++v) {
            GRBLinExpr expr = 0;
            for (int j = 0; j < n; ++j) expr += alpha[j][v];
            model.addConstr(y[v] <= expr, "nonempty_" + std::to_string(v));
        }

        for (int j = 0; j < n; ++j)
            for (int v = 0; v < m; ++v)
                model.addConstr(alpha[j][v] <= in.active[j], "inactive_" + std::to_string(j) + "_" + std::to_string(v));

        for (int r = 0; r < (int)in.cuts.size(); ++r) {
            const Stage3Cut& cut = in.cuts[r];
            GRBLinExpr expr = theta[cut.succ];
            for (int v = 0; v < m; ++v) {
                expr -= cut.piY[v] * y[v];
                for (int j = 0; j < n; ++j) expr -= cut.piAlpha[v][j] * alpha[j][v];
            }
            model.addConstr(expr >= cut.beta, "s3cut_" + std::to_string(r));
        }

        model.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);
        model.optimize();

        int status = model.get(GRB_IntAttr_Status);
        if (status != GRB_OPTIMAL) {
            sol.feasible = false;
            return sol;
        }

        sol.feasible = true;
        sol.obj = model.get(GRB_DoubleAttr_ObjVal);
        sol.alpha.assign(m, std::vector<int>(n, 0));
        sol.y.assign(m, 0);
        sol.z.assign(m, 0);
        sol.s.assign(n, 0);
        sol.theta.assign(in.numSucc, 0.0);

        for (int j = 0; j < n; ++j)
            for (int v = 0; v < m; ++v)
                sol.alpha[v][j] = (int)std::round(alpha[j][v].get(GRB_DoubleAttr_X));
        for (int j = 0; j < n; ++j) sol.s[j] = (int)std::round(s[j].get(GRB_DoubleAttr_X));
        for (int v = 0; v < m; ++v) sol.y[v] = (int)std::round(y[v].get(GRB_DoubleAttr_X));
        for (int v = 0; v < m; ++v) sol.z[v] = (int)std::round(z[v].get(GRB_DoubleAttr_X));
        for (int h = 0; h < in.numSucc; ++h) sol.theta[h] = theta[h].get(GRB_DoubleAttr_X);

    } catch (const GRBException& e) {
        throw std::runtime_error(std::string("Gurobi MIP error: ") + e.getMessage());
    }
    return sol;
}

} // namespace stage2bp

#ifdef STAGE2_BP_DEMO
#include <chrono>

namespace {

using namespace stage2bp;

struct TestResult {
    std::string name;
    bool pass = false;
    double bpObj = 0.0, mipObj = 0.0;
    double bpMs = 0.0, mipMs = 0.0;
};

void printSolution(const std::string& label, const Solution& sol, int n, int m) {
    std::cout << "  " << label << " obj = " << std::setprecision(10) << sol.obj << "\n";
    for (int v = 0; v < m; ++v) {
        std::cout << "    vehicle " << v << " (y=" << sol.y[v] << " z=" << sol.z[v] << "):";
        for (int j = 0; j < n; ++j) if (sol.alpha[v][j]) std::cout << " " << j;
        std::cout << "\n";
    }
    std::cout << "    outsourced:";
    for (int j = 0; j < n; ++j) if (sol.s[j]) std::cout << " " << j;
    std::cout << "\n";
    std::cout << "    theta:";
    for (double th : sol.theta) std::cout << " " << th;
    std::cout << "\n";
}

bool verifySolution(const Solution& sol, const Stage2Input& in) {
    if (!sol.feasible) return false;
    for (int j = 0; j < in.n; ++j) {
        int count = sol.s[j];
        for (int v = 0; v < in.m; ++v) count += sol.alpha[v][j];
        if (count != 1) { std::cerr << "  FAIL: fulfillment violated for j=" << j << "\n"; return false; }
    }
    for (int v = 0; v < in.m; ++v) {
        if (sol.y[v] > sol.z[v]) { std::cerr << "  FAIL: y>z for v=" << v << "\n"; return false; }
        double load = 0;
        int assigned = 0;
        for (int j = 0; j < in.n; ++j) {
            if (sol.alpha[v][j]) {
                load += in.volume[j];
                ++assigned;
                if (!in.active[j]) { std::cerr << "  FAIL: inactive j=" << j << " assigned to v=" << v << "\n"; return false; }
                if (!sol.y[v]) { std::cerr << "  FAIL: j=" << j << " assigned to inactive vehicle v=" << v << "\n"; return false; }
            }
        }
        if (load > in.Qv[v] + 1e-6) { std::cerr << "  FAIL: capacity exceeded for v=" << v << "\n"; return false; }
        if (sol.y[v] && assigned == 0) { std::cerr << "  FAIL: empty active vehicle v=" << v << "\n"; return false; }
    }
    for (const Stage3Cut& cut : in.cuts) {
        double lhs = sol.theta[cut.succ];
        double rhs = cut.beta;
        for (int v = 0; v < in.m; ++v) {
            rhs += cut.piY[v] * sol.y[v];
            for (int j = 0; j < in.n; ++j) rhs += cut.piAlpha[v][j] * sol.alpha[v][j];
        }
        if (lhs < rhs - 1e-6) {
            std::cerr << "  FAIL: theta[" << cut.succ << "]=" << lhs
                      << " < cut RHS=" << rhs << "\n";
            return false;
        }
    }
    for (int h = 0; h < in.numSucc; ++h) {
        if (sol.theta[h] < -1e-6) {
            std::cerr << "  FAIL: theta[" << h << "]=" << sol.theta[h] << " < 0\n";
            return false;
        }
    }
    return true;
}

TestResult runTest(const std::string& name, Stage2Input in, bool verbose = false) {
    TestResult tr;
    tr.name = name;
    std::cout << "=== " << name << " ===\n";

    SolverParams p;
    p.verbose = verbose;
    p.pricingTopK = 5;

    auto t0 = std::chrono::high_resolution_clock::now();
    Stage2BranchPriceSolver solver(in, p);
    Solution bpSol = solver.solve();
    auto t1 = std::chrono::high_resolution_clock::now();
    tr.bpMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto t2 = std::chrono::high_resolution_clock::now();
    Solution mipSol = solveGurobiMip(in);
    auto t3 = std::chrono::high_resolution_clock::now();
    tr.mipMs = std::chrono::duration<double, std::milli>(t3 - t2).count();

    tr.bpObj = bpSol.obj;
    tr.mipObj = mipSol.obj;

    printSolution("B&P", bpSol, in.n, in.m);
    printSolution("MIP", mipSol, in.n, in.m);

    bool bpValid = verifySolution(bpSol, in);
    bool mipValid = verifySolution(mipSol, in);

    double gap = std::fabs(tr.bpObj - tr.mipObj);
    tr.pass = bpSol.feasible && mipSol.feasible && bpValid && mipValid && gap < 1e-4;

    std::cout << "  B&P time: " << std::fixed << std::setprecision(1) << tr.bpMs << " ms"
              << "  MIP time: " << tr.mipMs << " ms"
              << "  gap: " << std::scientific << std::setprecision(2) << gap
              << "  " << (tr.pass ? "PASS" : "FAIL") << "\n\n";
    return tr;
}

Stage3Cut makeDummyCut(int succ, int m, int n) {
    Stage3Cut c;
    c.succ = succ;
    c.beta = 0.0;
    c.piY.assign(m, 0.0);
    c.piAlpha.assign(m, std::vector<double>(n, 0.0));
    return c;
}

} // anonymous namespace

int main() {
    using namespace stage2bp;
    std::vector<TestResult> results;

    // Test 1: Basic (5 customers, 2 vehicles, all active, trivial cut)
    {
        Stage2Input in;
        in.n = 5; in.m = 2; in.numSucc = 1;
        in.active = {1, 1, 1, 1, 1};
        in.volume = {2, 4, 3, 5, 2};
        in.cOut = {10, 9, 8, 11, 7};
        in.Qv = {7, 7};
        in.piZ = {0.0, 0.0};
        in.cuts.push_back(makeDummyCut(0, in.m, in.n));
        results.push_back(runTest("Test 1: Basic", in));
    }

    // Test 2: Inactive customers (7 customers, 3 inactive)
    {
        Stage2Input in;
        in.n = 7; in.m = 2; in.numSucc = 1;
        in.active = {1, 1, 0, 1, 0, 1, 0};
        in.volume = {3, 2, 4, 5, 1, 2, 3};
        in.cOut = {8, 6, 5, 12, 3, 7, 4};
        in.Qv = {8, 8};
        in.piZ = {0.0, 0.0};
        in.cuts.push_back(makeDummyCut(0, in.m, in.n));
        results.push_back(runTest("Test 2: Inactive customers", in));
    }

    // Test 3: Non-zero piZ (tests z elimination)
    {
        Stage2Input in;
        in.n = 5; in.m = 2; in.numSucc = 1;
        in.active = {1, 1, 1, 1, 1};
        in.volume = {2, 3, 4, 2, 3};
        in.cOut = {10, 8, 12, 9, 7};
        in.Qv = {7, 7};
        in.piZ = {3.0, -2.0};
        in.cuts.push_back(makeDummyCut(0, in.m, in.n));
        results.push_back(runTest("Test 3: Non-zero piZ", in));
    }

    // Test 4: Meaningful Stage 3 cuts
    {
        Stage2Input in;
        in.n = 4; in.m = 2; in.numSucc = 2;
        in.active = {1, 1, 1, 1};
        in.volume = {2, 3, 2, 3};
        in.cOut = {10, 10, 10, 10};
        in.Qv = {5, 5};
        in.piZ = {1.0, 1.0};

        Stage3Cut c1;
        c1.succ = 0; c1.beta = 5.0;
        c1.piY = {2.0, 1.0};
        c1.piAlpha = {{1.0, 0.5, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.5}};
        in.cuts.push_back(c1);

        Stage3Cut c2;
        c2.succ = 1; c2.beta = 3.0;
        c2.piY = {0.5, 1.5};
        c2.piAlpha = {{0.0, 1.0, 0.5, 0.0}, {0.5, 0.0, 0.0, 1.0}};
        in.cuts.push_back(c2);

        results.push_back(runTest("Test 4: Stage 3 cuts", in));
    }

    // Test 5: Outsourcing optimal (low capacity forces outsourcing)
    {
        Stage2Input in;
        in.n = 6; in.m = 2; in.numSucc = 1;
        in.active = {1, 1, 1, 1, 1, 1};
        in.volume = {3, 4, 5, 3, 4, 5};
        in.cOut = {2, 2, 2, 20, 20, 20};
        in.Qv = {6, 6};
        in.piZ = {0.0, 0.0};
        in.cuts.push_back(makeDummyCut(0, in.m, in.n));
        results.push_back(runTest("Test 5: Outsourcing optimal", in));
    }

    // Test 6: Larger instance (12 customers, 3 vehicles, multiple cuts)
    {
        Stage2Input in;
        in.n = 12; in.m = 3; in.numSucc = 2;
        in.active = {1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1};
        in.volume = {2, 3, 1, 4, 2, 3, 5, 2, 4, 1, 3, 2};
        in.cOut = {8, 12, 6, 15, 9, 10, 4, 7, 14, 5, 3, 11};
        in.Qv = {10, 8, 9};
        in.piZ = {2.0, -1.0, 0.5};

        Stage3Cut c1;
        c1.succ = 0; c1.beta = 4.0;
        c1.piY = {1.5, 0.5, 1.0};
        c1.piAlpha.assign(3, std::vector<double>(12, 0.0));
        c1.piAlpha[0][0] = 0.8; c1.piAlpha[0][1] = 0.3;
        c1.piAlpha[1][3] = 0.5; c1.piAlpha[1][4] = 0.7;
        c1.piAlpha[2][7] = 0.4; c1.piAlpha[2][8] = 0.6;
        in.cuts.push_back(c1);

        Stage3Cut c2;
        c2.succ = 1; c2.beta = 2.5;
        c2.piY = {0.8, 1.2, 0.3};
        c2.piAlpha.assign(3, std::vector<double>(12, 0.0));
        c2.piAlpha[0][2] = 0.6; c2.piAlpha[0][5] = 0.4;
        c2.piAlpha[1][1] = 0.9; c2.piAlpha[2][9] = 0.5;
        c2.piAlpha[2][11] = 0.3;
        in.cuts.push_back(c2);

        Stage3Cut c3;
        c3.succ = 0; c3.beta = 3.0;
        c3.piY = {0.5, 1.0, 0.8};
        c3.piAlpha.assign(3, std::vector<double>(12, 0.0));
        c3.piAlpha[0][4] = 0.3; c3.piAlpha[1][0] = 0.5;
        c3.piAlpha[1][8] = 0.4; c3.piAlpha[2][5] = 0.7;
        in.cuts.push_back(c3);

        results.push_back(runTest("Test 6: Larger instance", in));
    }

    // Summary
    std::cout << "==================== SUMMARY ====================\n";
    int passed = 0;
    for (const auto& r : results) {
        std::cout << (r.pass ? "PASS" : "FAIL") << "  " << r.name
                  << "  B&P=" << std::setprecision(6) << r.bpObj
                  << "  MIP=" << r.mipObj
                  << "  B&P_ms=" << std::fixed << std::setprecision(1) << r.bpMs
                  << "  MIP_ms=" << r.mipMs << "\n";
        if (r.pass) ++passed;
    }
    std::cout << passed << "/" << results.size() << " tests passed.\n";
    return (passed == (int)results.size()) ? 0 : 1;
}
#endif
