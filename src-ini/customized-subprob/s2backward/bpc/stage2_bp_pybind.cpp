/*
 * stage2_bp_pybind.cpp
 *
 * pybind11 wrapper exposing stage2bp::Stage2BranchPriceSolver to Python.
 *
 * Direct replacement for the Gurobi MIP solve in
 * `_solve_lagrangian_dual_s2` inner loop.  Given the current Lagrangian
 * multipliers piZ (one per vehicle), all S3->S2 cuts, and node data,
 * returns the exact Stage-2 Lagrangian subproblem optimum:
 *
 *   min  sum_h theta[h] + sum_j cOut[j]*s[j] - sum_v piZ[v]*z[v]
 *
 * Module name: stage2_bp_cpp
 * Function:    solve_stage2_lag(...)
 */

// Pull in the solver. main() is guarded by STAGE2_BP_DEMO; we do NOT
// define it here, so #include yields just the library code.
#include "stage2_branch_price.cpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <chrono>
#include <stdexcept>

namespace py = pybind11;

namespace {

// Convert Python list[float] / numpy 1d -> std::vector<double>.
inline std::vector<double> to_vec_d(const py::object& obj) {
    if (py::isinstance<py::array>(obj)) {
        auto arr = obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        const double* data = arr.data();
        return std::vector<double>(data, data + arr.size());
    }
    return obj.cast<std::vector<double>>();
}

inline std::vector<int> to_vec_i(const py::object& obj) {
    if (py::isinstance<py::array>(obj)) {
        auto arr = obj.cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
        const int* data = arr.data();
        return std::vector<int>(data, data + arr.size());
    }
    return obj.cast<std::vector<int>>();
}

inline std::vector<std::vector<double>> to_mat_d(const py::object& obj) {
    // Accept numpy (m, n) or list[list[float]].
    if (py::isinstance<py::array>(obj)) {
        auto arr = obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        if (arr.ndim() != 2) throw std::runtime_error("expected 2D array");
        const auto rows = static_cast<size_t>(arr.shape(0));
        const auto cols = static_cast<size_t>(arr.shape(1));
        const double* data = arr.data();
        std::vector<std::vector<double>> out(rows, std::vector<double>(cols));
        for (size_t i = 0; i < rows; ++i) {
            std::copy(data + i * cols, data + (i + 1) * cols, out[i].begin());
        }
        return out;
    }
    return obj.cast<std::vector<std::vector<double>>>();
}

// Build Stage3Cut list from a Python list of dicts:
//   { "succ": int, "beta": float, "piY": [m], "piAlpha": [[m x n]] }
std::vector<stage2bp::Stage3Cut> build_cuts(const py::list& py_cuts,
                                            int m, int n) {
    std::vector<stage2bp::Stage3Cut> out;
    out.reserve(py_cuts.size());
    for (const py::handle& h : py_cuts) {
        py::dict d = h.cast<py::dict>();
        stage2bp::Stage3Cut c;
        c.succ = d["succ"].cast<int>();
        c.beta = d["beta"].cast<double>();
        c.piY  = to_vec_d(d["piY"]);
        c.piAlpha = to_mat_d(d["piAlpha"]);
        if (static_cast<int>(c.piY.size()) != m) {
            throw std::runtime_error("piY size != m in cut");
        }
        if (static_cast<int>(c.piAlpha.size()) != m) {
            throw std::runtime_error("piAlpha rows != m in cut");
        }
        for (const auto& row : c.piAlpha) {
            if (static_cast<int>(row.size()) != n) {
                throw std::runtime_error("piAlpha cols != n in cut");
            }
        }
        out.emplace_back(std::move(c));
    }
    return out;
}

py::dict solve_stage2_lag(int n, int m, int numSucc,
                          py::object active,
                          py::object volume,
                          py::object cOut,
                          py::object Qv,
                          py::object piZ,
                          py::list cuts,
                          int pricing_top_k,
                          int max_nodes,
                          int max_depth,
                          int max_colgen_iters,
                          double rc_tol,
                          double int_tol,
                          double time_limit_s,
                          bool verbose,
                          bool use_heuristic_pricing,
                          bool use_vehicle_clustering,
                          bool use_diving,
                          bool use_ryan_foster,
                          bool use_dual_stabilization,
                          double cull_rc_threshold,
                          int num_threads,
                          int pricing_top_k_root,
                          int pricing_top_k_shallow,
                          int pricing_top_k_deep,
                          bool use_restricted_mip,
                          double restricted_mip_time_limit,
                          bool use_cut_aging,
                          double theta_lower_bound,
                          int solve_mode,
                          double forward_gap,
                          bool root_bound_only,
                          int max_cutting_rounds,
                          bool use_sr3_cuts,
                          bool use_cover_cuts,
                          bool use_clique_cuts,
                          py::object vehicle_types,
                          bool use_purchase_order,
                          double forward_reference_lb,
                          double backward_gap_abs,
                          double backward_gap_rel) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();

    // Validate raw scalar arguments before a solve-mode preset can replace or
    // ignore them.  In particular, NaN compares false in expressions such as
    // `forward_gap > 0`, so relying only on the solver-side parameter check
    // would silently turn a malformed native payload into the disabled value.
    if (!std::isfinite(rc_tol) || rc_tol < 0.0)
        throw std::invalid_argument("rc_tol must be finite and nonnegative");
    if (!std::isfinite(int_tol) || int_tol < 0.0 || int_tol >= 0.5)
        throw std::invalid_argument("int_tol must be finite and in [0, 0.5)");
    if (!std::isfinite(time_limit_s))
        throw std::invalid_argument("time_limit_s must be finite");
    if (!std::isfinite(cull_rc_threshold))
        throw std::invalid_argument("cull_rc_threshold must be finite");
    if (!std::isfinite(restricted_mip_time_limit)
            || restricted_mip_time_limit < 0.0)
        throw std::invalid_argument(
            "restricted_mip_time_limit must be finite and nonnegative");
    if (!std::isfinite(theta_lower_bound))
        throw std::invalid_argument("theta_lower_bound must be finite");
    if (!std::isfinite(forward_gap))
        throw std::invalid_argument("forward_gap must be finite");
    if (!std::isfinite(backward_gap_abs) || backward_gap_abs < 0.0
            || !std::isfinite(backward_gap_rel) || backward_gap_rel < 0.0)
        throw std::invalid_argument("backward gaps must be finite and nonnegative");
    if ((backward_gap_abs > 0.0 || backward_gap_rel > 0.0) && solve_mode != 2)
        throw std::invalid_argument("backward gaps require solve_mode=2");
    if (solve_mode < 0 || solve_mode > 2)
        throw std::invalid_argument("solve_mode must be 0, 1, or 2");
    if (!std::isfinite(forward_reference_lb))
        throw std::invalid_argument("forward_reference_lb must be finite");
    if (forward_reference_lb > -1e100 && solve_mode != 1)
        throw std::invalid_argument("forward_reference_lb requires solve_mode=1");

    stage2bp::Stage2Input in;
    in.n = n;
    in.m = m;
    in.numSucc = numSucc;
    in.active = to_vec_i(active);
    in.volume = to_vec_d(volume);
    in.cOut   = to_vec_d(cOut);
    in.Qv     = to_vec_d(Qv);
    in.piZ    = to_vec_d(piZ);
    if (!vehicle_types.is_none()) in.vehicleType = to_vec_i(vehicle_types);

    if (static_cast<int>(in.active.size()) != n) throw std::runtime_error("active size != n");
    if (static_cast<int>(in.volume.size()) != n) throw std::runtime_error("volume size != n");
    if (static_cast<int>(in.cOut.size())   != n) throw std::runtime_error("cOut size != n");
    if (static_cast<int>(in.Qv.size())     != m) throw std::runtime_error("Qv size != m");
    if (static_cast<int>(in.piZ.size())    != m) throw std::runtime_error("piZ size != m");

    in.cuts = build_cuts(cuts, m, n);

    stage2bp::SolverParams params;
    params.pricingTopK     = pricing_top_k;
    params.maxNodes        = max_nodes;
    params.maxDepth        = max_depth;
    params.maxColgenIters  = max_colgen_iters;
    params.rcTol           = rc_tol;
    params.intTol          = int_tol;
    params.timeLimitSec    = time_limit_s;
    params.verbose         = verbose;
    params.useHeuristicPricing  = use_heuristic_pricing;
    params.useVehicleClustering = use_vehicle_clustering;
    params.useDiving            = use_diving;
    params.useRyanFoster        = use_ryan_foster;
    params.useDualStabilization = use_dual_stabilization;
    params.cullRcThreshold      = cull_rc_threshold;
    params.numThreads           = num_threads;
    params.pricingTopKRoot      = pricing_top_k_root;
    params.pricingTopKShallow   = pricing_top_k_shallow;
    params.pricingTopKDeep      = pricing_top_k_deep;
    params.useRestrictedMip     = use_restricted_mip;
    params.restrictedMipTimeLimit = restricted_mip_time_limit;
    params.useCutAging          = use_cut_aging;
    params.thetaLowerBound      = theta_lower_bound;
    params.rootBoundOnly        = root_bound_only;
    params.backwardGapAbs       = backward_gap_abs;
    params.backwardGapRel       = backward_gap_rel;
    params.useSR3Cuts           = use_sr3_cuts;
    params.useCoverCuts         = use_cover_cuts;
    params.useCliqueCuts        = use_clique_cuts;
    params.usePurchaseOrder     = use_purchase_order;

    // ------------------------------------------------------------------
    // solve_mode preset (forward vs backward split).
    //   0 = balanced / legacy (no change; fully backward-compatible)
    //   1 = UB  (forward): pour effort into primal heuristics so the best
    //           feasible incumbent is found fast. Cuts only tighten the LB /
    //           prove optimality, which forward does not need; the incumbent
    //           is always validated by fullFeasibilityCheck, so the returned
    //           UB stays valid regardless.
    //   2 = LB  (backward): the returned lb == root LP-after-cuts bound, so
    //           pour effort into VALID cutting at the root to tighten it. More
    //           cuts can only raise the LP bound while keeping it valid; the
    //           UB-only restricted-MIP heuristic is dropped to save time. The
    //           historical full B&P tree remains the default; root_bound_only
    //           must be requested explicitly.
    // ------------------------------------------------------------------
    if (solve_mode == 1) {
        params.useDiving = true;
        if (params.divingMaxDepth < 50) params.divingMaxDepth = 50;
        params.useRestrictedMip = true;
        params.restrictedMipFreq = 1;
        if (params.restrictedMipTimeLimit < 1.0) params.restrictedMipTimeLimit = 1.0;
        // Physical LRP facilities are distinct when the caller disables
        // both clustering and purchase-order symmetry. Keep the existing
        // alpha-primary / Ryan-Foster-fallback tree for that domain; suppressing
        // alpha after a missing pair is justified only for the legacy fleet.
        const bool distinctFacilities = !use_vehicle_clustering && !use_purchase_order;
        params.preferRyanFoster = !distinctFacilities;
        // Forward fast-UB behaviour: heuristic-only CG at depth>0, root-LB+gap
        // global early stop. The no-alpha symmetry closure is legacy-only.
        params.forwardUbMode = true;
        params.forwardReferenceLb = forward_reference_lb;
        // MIPGap-style early stop: forward only needs a good, valid UB.
        if (forward_gap > 0.0) params.forwardGap = forward_gap;

        // --- P0: forward = fast valid UB, not a tight LP / optimality proof ---
        // Cuts only tighten the LB, which forward does not need: keep at most one
        // cheap separation round and drop clique cuts entirely.
        params.maxCuttingRounds = 1;
        if (params.maxSR3Cuts   > 30) params.maxSR3Cuts   = 30;
        if (params.maxCoverCuts > 30) params.maxCoverCuts = 30;
        params.maxCliqueCuts = 0;
        if (params.maxSR3PerRound   > 15) params.maxSR3PerRound   = 15;
        if (params.maxCoverPerRound > 15) params.maxCoverPerRound = 15;
        // CG effort cap per node (heuristic pricing dominates the UB anyway).
        if (params.maxColgenIters > 300) params.maxColgenIters = 300;
        // Strong branching only ranks candidates to tighten the proof: skip it.
        params.strongBranchMaxDepth = 0;
        // piZ≈0 in forward, so Wentges smoothing buys little; raw duals are fine.
        params.useDualStabilization = false;
        // B&B size guards: identical fleets can blow up the tree. Tighten only
        // if the caller did not already pass something smaller.
        if (params.maxDepth > 60)    params.maxDepth = 60;
        if (params.maxNodes > 20000) params.maxNodes = 20000;
    } else if (solve_mode == 2) {
        if (params.maxCuttingRounds < 30) params.maxCuttingRounds = 30;
        if (params.maxSR3Cuts       < 500) params.maxSR3Cuts       = 500;
        if (params.maxCoverCuts     < 300) params.maxCoverCuts     = 300;
        if (params.maxCliqueCuts    < 50)  params.maxCliqueCuts    = 50;
        if (params.maxSR3PerRound   < 60)  params.maxSR3PerRound   = 60;
        if (params.maxCoverPerRound < 30)  params.maxCoverPerRound = 30;
        params.useRestrictedMip = false;
    }

    // Explicit caller settings override the solve-mode preset.  -1 preserves
    // the historical preset/default, while 0 means root CG with no separation
    // round and N>0 allows at most N separation/re-pricing rounds.
    if (max_cutting_rounds < -1) {
        throw std::invalid_argument("max_cutting_rounds must be -1 or >= 0");
    }
    if (max_cutting_rounds >= 0) {
        params.maxCuttingRounds = max_cutting_rounds;
    }
    if (root_bound_only && solve_mode != 2) {
        throw std::invalid_argument(
            "root_bound_only is a backward/LB option and requires solve_mode=2");
    }

    stage2bp::Stage2BranchPriceSolver solver(in, params);
    stage2bp::Solution sol = solver.solve();

    auto t1 = clk::now();
    double t_solve = std::chrono::duration<double>(t1 - t0).count();

    py::dict out;
    out["feasible"]        = sol.feasible;
    out["obj"]             = sol.feasible ? sol.obj : std::numeric_limits<double>::infinity();
    // `lb_certified` means the returned LB is mathematically valid.
    // `optimality_proven` is the stronger statement lb == incumbent obj.
    // A later child timeout does not invalidate an exact root-LP bound.
    out["lb"]              = sol.lb;
    out["timed_out"]       = sol.timed_out;
    out["nodes_processed"] = sol.nodes_processed;
    out["t_solve"]         = t_solve;

    // Certification flags.
    out["lb_certified"]     = sol.lb_certified;
    out["ub_certified"]     = sol.ub_certified;
    out["optimality_proven"] = sol.optimality_proven;
    out["tree_complete"]     = sol.tree_complete;
    out["intentional_root_stop"] = sol.intentional_root_stop;
    out["proof_relaxed"] = sol.proof_relaxed;
    out["tolerance_bound_prunes"] = sol.tolerance_bound_prunes;
    out["tolerance_integral_closures"] = sol.tolerance_integral_closures;
    out["pricing_certified_nodes"]   = sol.pricing_certified_nodes;
    out["pricing_uncertified_nodes"] = sol.pricing_uncertified_nodes;
    out["used_inherited_lb_nodes"]   = sol.used_inherited_lb_nodes;

    // Instrumentation breakdown (seconds and counters).
    out["total_time_s"]     = sol.total_time_s;
    out["rmp_build_time_s"] = sol.rmp_build_time_s;
    out["rmp_solve_time_s"] = sol.rmp_solve_time_s;
    out["pricing_time_s"]   = sol.pricing_time_s;
    out["cut_separation_time_s"] = sol.cut_separation_time_s;
    out["primal_heuristic_time_s"] = sol.primal_heuristic_time_s;
    out["forward_cg_primal_calls"] = sol.forward_cg_primal_calls;
    out["forward_cg_primal_improvements"] = sol.forward_cg_primal_improvements;
    out["forward_reference_stop"] = sol.forward_reference_stop;
    out["backward_gap_stop"] = sol.backward_gap_stop;
    out["forward_rounding_calls"] = sol.forward_rounding_calls;
    out["forward_rounding_improvements"] = sol.forward_rounding_improvements;
    out["forward_partial_bound_calls"] = sol.forward_partial_bound_calls;
    out["forward_partial_bound_certificates"] = sol.forward_partial_bound_certificates;
    out["forward_approximate_cg_returns"] = sol.forward_approximate_cg_returns;
    out["cg_iters_total"]   = sol.cg_iters_total;
    out["lp_solves"]        = sol.lp_solves;
    out["columns_generated"] = sol.columns_generated;
    out["exact_pricing_calls"] = sol.exact_pricing_calls;
    out["pricing_deadline_interruptions"] = sol.pricing_deadline_interruptions;
    out["heuristic_pricing_successes"] = sol.heuristic_pricing_successes;

    // Root node LP statistics.
    out["root_lp_before_cuts"] = sol.root_lp_before_cuts;
    out["root_lp_after_cuts"]  = sol.root_lp_after_cuts;
    out["root_columns"]        = sol.root_columns;
    out["root_cg_iters"]       = sol.root_cg_iters;
    out["root_pricing_passes_completed"] = sol.root_pricing_passes_completed;
    out["root_cut_rounds_requested"] = sol.root_cut_rounds_requested;
    out["root_cut_rounds_completed"] = sol.root_cut_rounds_completed;

    // Cut pool statistics.
    out["num_sr3_active"]    = sol.num_sr3_active;
    out["num_cover_active"]  = sol.num_cover_active;
    out["num_clique_active"] = sol.num_clique_active;
    out["root_bound_only"]    = sol.root_bound_only;
    out["sr3_cuts_enabled"]   = sol.sr3_cuts_enabled;
    out["cover_cuts_enabled"] = sol.cover_cuts_enabled;
    out["clique_cuts_enabled"] = sol.clique_cuts_enabled;
    out["purchase_order_enabled"] = sol.purchase_order_enabled;

    // Why the solve stopped early (empty = clean finish). Non-empty means an
    // internal exception was swallowed as a soft timeout.
    out["abort_reason"] = sol.abort_reason;
    if (!sol.abort_reason.empty()) {
        out["termination_reason"] = "internal_error";
    } else if (sol.optimality_proven) {
        out["termination_reason"] = "optimal";
    } else if (sol.backward_gap_stop) {
        out["termination_reason"] = "backward_query_gap";
    } else if (sol.intentional_root_stop) {
        out["termination_reason"] = "root_bound_only";
    } else if (sol.forward_reference_stop) {
        out["termination_reason"] = "forward_target";
    } else if (sol.timed_out) {
        out["termination_reason"] = "resource_limit";
    } else {
        out["termination_reason"] = "incomplete";
    }

    if (!sol.feasible) {
        out["alpha"] = py::list();
        out["y"]     = py::list();
        out["z"]     = py::list();
        out["s"]     = py::list();
        out["theta"] = py::list();
        return out;
    }

    py::list py_alpha;
    for (int v = 0; v < m; ++v) {
        py::list row;
        for (int j = 0; j < n; ++j) row.append(sol.alpha[v][j]);
        py_alpha.append(std::move(row));
    }
    out["alpha"] = py_alpha;
    out["y"]     = py::cast(sol.y);
    out["z"]     = py::cast(sol.z);
    out["s"]     = py::cast(sol.s);
    out["theta"] = py::cast(sol.theta);

    return out;
}

}  // namespace

PYBIND11_MODULE(stage2_bp_cpp, m) {
    m.attr("backward_deadline_contract") = "all_mode_pricing_deadline_v1";
    m.attr("backward_gap_contract") = "certified_root_query_gap_v1";
    m.attr("forward_anytime_contract") = "lrp_forward_anytime_v1";
    m.doc() = "Stage-2 Lagrangian assignment subproblem via exact branch-and-price.\n"
              "Drop-in replacement for the Gurobi MIP solve in "
              "_solve_lagrangian_dual_s2 inner loop.";

    m.def("solve_stage2_lag", &solve_stage2_lag,
          py::arg("n"),
          py::arg("m"),
          py::arg("numSucc"),
          py::arg("active"),
          py::arg("volume"),
          py::arg("cOut"),
          py::arg("Qv"),
          py::arg("piZ"),
          py::arg("cuts"),
          py::arg("pricing_top_k")    = 5,
          py::arg("max_nodes")        = 1000000,
          py::arg("max_depth")        = 100000,
          py::arg("max_colgen_iters") = 1000,
          py::arg("rc_tol")           = 1e-7,
          py::arg("int_tol")          = 1e-6,
          py::arg("time_limit_s")     = -1.0,
          py::arg("verbose")          = false,
          py::arg("use_heuristic_pricing")  = true,
          py::arg("use_vehicle_clustering") = true,
          py::arg("use_diving")             = true,
          py::arg("use_ryan_foster")        = true,
          py::arg("use_dual_stabilization") = true,
          py::arg("cull_rc_threshold")      = 10.0,
          py::arg("num_threads")            = 1,
          py::arg("pricing_top_k_root")   = 30,
          py::arg("pricing_top_k_shallow") = 20,
          py::arg("pricing_top_k_deep")   = 5,
          py::arg("use_restricted_mip")   = true,
          py::arg("restricted_mip_time_limit") = 0.5,
          py::arg("use_cut_aging")        = true,
          py::arg("theta_lower_bound")    = 0.0,
          py::arg("solve_mode")           = 0,
          py::arg("forward_gap")          = -1.0,
          py::arg("root_bound_only")      = false,
          py::arg("max_cutting_rounds")   = -1,
          py::arg("use_sr3_cuts")         = true,
          py::arg("use_cover_cuts")       = true,
          py::arg("use_clique_cuts")      = true,
          py::arg("vehicle_types")        = py::none(),
          py::arg("use_purchase_order")   = false,
          py::arg("forward_reference_lb") = -1e100,
          py::arg("backward_gap_abs") = 0.0,
          py::arg("backward_gap_rel") = 0.0,
R"pbdoc(
Exact branch-and-price for the Stage-2 Lagrangian assignment subproblem.

Objective (minimize):
    sum_h theta[h] + sum_j cOut[j] * s[j] - sum_v piZ[v] * z[v]

subject to standard Stage-2 constraints (assignment, capacity, etc.) and
the S3->S2 Benders/Lagrangian cuts.

Args
----
n, m, numSucc : int
    Number of customers, vehicles, and successor (theta) variables.
active : list[int] or 1D array of size n  (0/1)
volume : list[float] or 1D array of size n
cOut   : list[float] or 1D array of size n
Qv     : list[float] or 1D array of size m
piZ    : list[float] or 1D array of size m
    Current Lagrangian multipliers pi_value["z[v,t]"].
cuts   : list of dict
    Each cut: {"succ": int, "beta": float, "piY": [m], "piAlpha": [[m x n]]}
    Row form: theta[succ] - sum_{v,S} a(cut,v,S) lambda[v,S] >= beta
pricing_top_k : int, default 5
    Top-K columns added per CG iteration.

time_limit_s : float, default -1.0
    Wall-clock soft cap for BP. <=0 disables. When triggered, BP returns the
    current greedy/incumbent (if any) and a certified root LP relaxation LB
    when exact root pricing completed.  Check `lb_certified` before using `lb`.
root_bound_only : bool, default False
    Backward-only mode (requires solve_mode=2): return immediately after the
    configured root cut-and-price rounds have been exactly priced.
max_cutting_rounds : int, default -1
    -1 keeps the solve-mode preset.  0 disables separation; N>0 allows at most
    N root separation/re-pricing rounds.
vehicle_types : list[int] of size m, optional
    Canonical type id for each vehicle. Required when use_purchase_order=True.
use_purchase_order : bool, default False
    In root-bound-only backward mode, add the exact leading-prefix purchase
    block and include its link/prefix duals in the certified root bound.

Returns
-------
dict with keys:
    feasible        : bool
    obj             : float    -- incumbent objective (UB on subproblem optimum)
    lb              : float    -- root LP LB, or obj when optimality is proven.
    lb_certified    : bool     -- True iff `lb` is a valid mathematical LB.
    ub_certified    : bool     -- True iff `obj` is a fully checked incumbent.
    optimality_proven : bool   -- True iff the complete exact tree proves
                                  lb == obj.  This, not merely timed_out=False,
                                  is the exactness flag.
    tree_complete    : bool     -- True iff the complete tree was fathomed.
    intentional_root_stop : bool -- True iff root-only mode deliberately
                                    skipped the child tree.
    proof_relaxed    : bool     -- True iff a positive tolerance/gap closed a
                                  node; then optimality_proven is always False.
    timed_out       : bool     -- True if wall-clock / maxNodes / maxDepth /
                                  maxColgen was hit before proving optimality.
    nodes_processed : int      -- BB nodes explored
    alpha           : list[list[int]]  m x n binary
    y               : list[int]  m
    z               : list[int]  m       (used for subgrad on pi_value["z[v,t]"])
    s               : list[int]  n
    theta           : list[float] numSucc
    t_solve         : float (seconds)
)pbdoc");
}
