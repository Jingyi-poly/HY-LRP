// Subset-DP kernels for the Stage-2 fixed-fleet piece solver.
//
// Mirrors s2forward/subset_dp.py exactly (same recurrences, same ordering
// rule, same chain-copy trace); Python keeps the per-node precomputation, the
// loop over vehicle types and the exact certification.
//
//   theta_table(coeffs[K][n], intercept_used[K], intercept_empty[K], feasible[2^n])
//       -> theta[2^n]   (max(0, best cut) per set, +inf when infeasible)
//   type_layer(g[2^n], thetas[p][2^n], masks_desc[F], group_starts[G+1], tail[p+1])
//       -> (g_new[2^n], slot_sets[p][2^n])
//
// Build: see build.sh next to this file (pybind11, -O3).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

py::array_t<double> theta_table(
    py::array_t<double, py::array::c_style | py::array::forcecast> coeffs,
    py::array_t<double, py::array::c_style | py::array::forcecast> intercept_used,
    py::array_t<double, py::array::c_style | py::array::forcecast> intercept_empty,
    py::array_t<bool, py::array::c_style | py::array::forcecast> feasible) {
    if (coeffs.ndim() != 2) throw std::invalid_argument("coeffs must be K x n");
    const py::ssize_t K = coeffs.shape(0);
    const py::ssize_t n = coeffs.shape(1);
    const py::ssize_t size = py::ssize_t(1) << n;
    if (feasible.size() != size) throw std::invalid_argument("feasible must have 2^n entries");
    if (intercept_used.size() != K || intercept_empty.size() != K)
        throw std::invalid_argument("intercepts must have K entries");

    py::array_t<double> out(size);
    double* theta = out.mutable_data();
    const bool* feas = feasible.data();
    const double* c = coeffs.data();
    const double* bu = intercept_used.data();
    const double* be = intercept_empty.data();
    {
        // Each vehicle owns an independent theta table.  Releasing the GIL
        // lets Python build those tables concurrently without changing this
        // exhaustive calculation or its deterministic per-table order.
        py::gil_scoped_release release;
        std::vector<double> sums(size);
        for (py::ssize_t s = 0; s < size; ++s) theta[s] = 0.0;

        std::vector<py::ssize_t> feasible_masks;
        feasible_masks.reserve(size);
        bool downward_closed = true;
        for (py::ssize_t s = 1; s < size; ++s) {
            if (!feas[s]) continue;
            feasible_masks.push_back(s);
            const py::ssize_t predecessor = s & (s - 1);
            if (predecessor != 0 && !feas[predecessor]) downward_closed = false;
        }
        // The compact feasible scan wins while it skips at least one fifth
        // of the table; above that point the fully sequential scan is faster.
        const bool sparse = downward_closed && feasible_masks.size() * 5 <
                                                static_cast<std::size_t>(size) * 4;

        for (py::ssize_t k = 0; k < K; ++k) {
            const double base = bu[k];
            sums[0] = 0.0;
            if (sparse) {
                for (const py::ssize_t s : feasible_masks) {
                    const int bit = __builtin_ctzll(static_cast<unsigned long long>(s));
                    sums[s] = sums[s & (s - 1)] + c[k * n + bit];
                    const double rhs = base + sums[s];
                    if (rhs > theta[s]) theta[s] = rhs;
                }
            } else {
                // Subset sums of this cut's coefficients by lowest-bit recursion.
                for (py::ssize_t s = 1; s < size; ++s) {
                    const int bit = __builtin_ctzll(static_cast<unsigned long long>(s));
                    sums[s] = sums[s & (s - 1)] + c[k * n + bit];
                    const double rhs = base + sums[s];
                    if (rhs > theta[s]) theta[s] = rhs;
                }
            }
        }
        for (py::ssize_t s = 1; s < size; ++s) {
            if (!feas[s]) theta[s] = kInf;
        }
        double empty = 0.0;
        for (py::ssize_t k = 0; k < K; ++k) empty = std::max(empty, be[k]);
        theta[0] = empty;
    }
    return out;
}

py::tuple price_table(
    py::array_t<double, py::array::c_style | py::array::forcecast> theta_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> item_coeff,
    double nonempty_coeff,
    double constant,
    int top_k,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> pair_masks,
    py::array_t<double, py::array::c_style | py::array::forcecast> pair_coeff,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> apart_masks) {
    if (theta_in.ndim() != 1) throw std::invalid_argument("theta must be one-dimensional");
    if (item_coeff.ndim() != 1) throw std::invalid_argument("item_coeff must be one-dimensional");
    const py::ssize_t n = item_coeff.size();
    if (n < 0 || n >= 63) throw std::invalid_argument("unsupported item count");
    const py::ssize_t size = py::ssize_t(1) << n;
    if (theta_in.size() != size) throw std::invalid_argument("theta must have 2^n entries");
    top_k = std::max(1, top_k);

    const double* theta = theta_in.data();
    const double* coeff = item_coeff.data();
    if (pair_masks.size() != pair_coeff.size())
        throw std::invalid_argument("pair_masks and pair_coeff must have equal length");
    const int64_t* pairs = pair_masks.data();
    const double* pair_cost = pair_coeff.data();
    const int64_t* apart = apart_masks.data();
    const py::ssize_t n_pairs = pair_masks.size();
    const py::ssize_t n_apart = apart_masks.size();
    // Keep the best few values in ascending order.  K is deliberately small
    // (normally 8--32), so insertion is faster than sorting all 2^n masks.
    std::vector<std::pair<double, int64_t>> best;
    best.reserve(static_cast<std::size_t>(top_k));
    {
        // Vehicle pricings are independent.  Release the GIL so the Python
        // driver can scan several private theta tables concurrently.
        py::gil_scoped_release release;
        std::vector<double> sums(size, 0.0);
        for (py::ssize_t mask = 0; mask < size; ++mask) {
            if (mask) {
                const int bit = __builtin_ctzll(static_cast<unsigned long long>(mask));
                sums[mask] = sums[mask & (mask - 1)] + coeff[bit];
            }
            if (!std::isfinite(theta[mask])) continue;
            bool branch_feasible = true;
            for (py::ssize_t i = 0; i < n_apart; ++i) {
                if ((static_cast<int64_t>(mask) & apart[i]) == apart[i]) {
                    branch_feasible = false;
                    break;
                }
            }
            if (!branch_feasible) continue;
            double value = theta[mask] + sums[mask]
                         + (mask ? nonempty_coeff : 0.0) + constant;
            for (py::ssize_t i = 0; i < n_pairs; ++i) {
                if ((static_cast<int64_t>(mask) & pairs[i]) == pairs[i])
                    value += pair_cost[i];
            }
            if (static_cast<int>(best.size()) == top_k && value >= best.back().first)
                continue;
            auto where = std::lower_bound(
                best.begin(), best.end(), value,
                [](const std::pair<double, int64_t>& item, double target) {
                    return item.first < target;
                });
            best.insert(where, {value, static_cast<int64_t>(mask)});
            if (static_cast<int>(best.size()) > top_k) best.pop_back();
        }
    }
    if (best.empty()) throw std::runtime_error("pricing table has no feasible mask");

    py::array_t<int64_t> masks(best.size());
    py::array_t<double> values(best.size());
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(best.size()); ++i) {
        values.mutable_at(i) = best[i].first;
        masks.mutable_at(i) = best[i].second;
    }
    // The exhaustive combinatorial search is complete.  Leave a conservative
    // binary64 accumulation margin before this value enters a dual repair.
    const double optimum = best.front().first;
    const double slack = 1e-10 + 1e-12 * std::abs(optimum);
    const double lower_bound = std::nextafter(optimum - slack,
                                               -std::numeric_limits<double>::infinity());
    return py::make_tuple(optimum, lower_bound, masks, values);
}

py::tuple terminal_single_layer(
    py::array_t<double, py::array::c_style | py::array::forcecast> g_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> theta_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> subset_reward,
    py::array_t<double, py::array::c_style | py::array::forcecast> tail) {
    if (g_in.ndim() != 1 || theta_in.ndim() != 1 || subset_reward.ndim() != 1)
        throw std::invalid_argument("terminal arrays must be one-dimensional");
    const py::ssize_t size = g_in.size();
    if (size < 1 || (size & (size - 1)) != 0)
        throw std::invalid_argument("terminal arrays must have power-of-two length");
    if (theta_in.size() != size || subset_reward.size() != size)
        throw std::invalid_argument("terminal arrays must have equal length");
    if (tail.size() != 2) throw std::invalid_argument("terminal tail must have two entries");

    const double* g = g_in.data();
    const double* theta = theta_in.data();
    const double* reward = subset_reward.data();
    const double* tail_cost = tail.data();
    const int64_t full = static_cast<int64_t>(size) - 1;
    double objective;
    int32_t previous_mask;
    int32_t assigned_mask;
    {
        py::gil_scoped_release release;
        std::vector<double> best(size);
        std::vector<int32_t> arg(size);
        for (int64_t mask = 0; mask < static_cast<int64_t>(size); ++mask) {
            best[mask] = g[mask] - reward[mask];
            arg[mask] = static_cast<int32_t>(mask);
        }
        for (int64_t step = 1; step < static_cast<int64_t>(size); step <<= 1) {
            const int64_t block_size = step << 1;
            for (int64_t block = 0; block < static_cast<int64_t>(size);
                 block += block_size) {
                for (int64_t offset = 0; offset < step; ++offset) {
                    const int64_t source = block + offset;
                    const int64_t target = source + step;
                    if (best[source] < best[target]) {
                        best[target] = best[source];
                        arg[target] = arg[source];
                    }
                }
            }
        }

        double reduced = best[full] + tail_cost[1];
        previous_mask = arg[full];
        assigned_mask = 0;
        for (int64_t mask = 1; mask < static_cast<int64_t>(size); ++mask) {
            if (!std::isfinite(theta[mask])) continue;
            const int64_t complement = full ^ mask;
            const double candidate = best[complement] + theta[mask]
                                   - reward[mask] + tail_cost[0];
            if (candidate < reduced) {
                reduced = candidate;
                previous_mask = arg[complement];
                assigned_mask = static_cast<int32_t>(mask);
            }
        }
        objective = reward[full] + reduced;
    }
    return py::make_tuple(objective, previous_mask, assigned_mask);
}

py::tuple type_layer(
    py::array_t<double, py::array::c_style | py::array::forcecast> g_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> thetas,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> masks_desc,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_starts,
    py::array_t<double, py::array::c_style | py::array::forcecast> tail) {
    if (thetas.ndim() != 2) throw std::invalid_argument("thetas must be p x 2^n");
    const py::ssize_t p = thetas.shape(0);
    const py::ssize_t size = thetas.shape(1);
    if (g_in.size() != size) throw std::invalid_argument("g must have 2^n entries");
    if (tail.size() != p + 1) throw std::invalid_argument("tail must have p+1 entries");
    if (p < 1) throw std::invalid_argument("p must be >= 1");
    const int64_t full = static_cast<int64_t>(size) - 1;

    // f[i]: value with i slots used.  chain[i] stores its i slot masks in
    // one contiguous, slot-major block.  The old vector-of-vectors layout
    // allocated and zeroed one unused size-entry block for every i and added
    // an extra pointer chase to every trace copy.
    std::vector<std::vector<double>> f(p + 1, std::vector<double>(size, kInf));
    std::copy(g_in.data(), g_in.data() + size, f[0].begin());
    std::vector<std::vector<int32_t>> chain(p + 1);
    for (py::ssize_t i = 1; i <= p; ++i)
        chain[i].assign(i * size, 0);

    const int64_t* masks = masks_desc.data();
    const int64_t* starts = group_starts.data();
    const py::ssize_t n_groups = group_starts.size() - 1;
    const int64_t n_masks = starts[n_groups];
    const double* th = thetas.data();

    // Slot 0 reads f[0] = g_in, which this layer never modifies.  When g_in
    // has few finite states (the first type: only the empty set) iterating
    // "finite T, then masks disjoint from T" is far cheaper than enumerating
    // every subset of the complement of every mask; both produce exactly the
    // same set of (T, mask) updates in the same group/mask order.
    std::vector<int64_t> finite_src0;
    {
        double dense_steps = 0.0;
        const double* theta0 = th;
        for (int64_t mi = 0; mi < n_masks; ++mi)
            if (std::isfinite(theta0[masks[mi]]))
                dense_steps += std::ldexp(1.0, __builtin_popcountll(
                    static_cast<unsigned long long>(full ^ masks[mi])));
        for (int64_t T = 0; T < static_cast<int64_t>(size); ++T)
            if (std::isfinite(f[0][T])) finite_src0.push_back(T);
        const double sparse_steps =
            static_cast<double>(finite_src0.size()) * static_cast<double>(n_masks);
        if (!(sparse_steps < dense_steps)) finite_src0.clear();  // dense path
    }
    const bool sparse_slot0 = !finite_src0.empty();

    for (py::ssize_t gi = 0; gi < n_groups; ++gi) {
        const int64_t lo = starts[gi];
        const int64_t hi = starts[gi + 1];
        for (py::ssize_t slot = 0; slot < p; ++slot) {
            const double* theta = th + slot * size;
            std::vector<double>& src = f[slot];
            std::vector<double>& dst = f[slot + 1];
            std::vector<int32_t>& dst_chain = chain[slot + 1];
            if (slot == 0 && sparse_slot0) {
                for (int64_t mi = lo; mi < hi; ++mi) {
                    const int64_t mask = masks[mi];
                    const double cost = theta[mask];
                    if (!std::isfinite(cost)) continue;
                    for (const int64_t T : finite_src0) {
                        if (T & mask) continue;
                        const double cand = src[T] + cost;
                        const int64_t R1 = T | mask;
                        if (cand < dst[R1]) {
                            dst[R1] = cand;
                            dst_chain[R1] = static_cast<int32_t>(mask);
                        }
                    }
                }
                continue;
            }
            for (int64_t mi = lo; mi < hi; ++mi) {
                const int64_t mask = masks[mi];
                const double cost = theta[mask];
                if (!std::isfinite(cost)) continue;
                const int64_t comp = full ^ mask;
                // All T subset of comp: states without the customers in mask.
                int64_t T = comp;
                while (true) {
                    const double cand = src[T] + cost;
                    const int64_t R1 = T | mask;
                    if (cand < dst[R1]) {
                        dst[R1] = cand;
                        for (py::ssize_t k = 0; k < slot; ++k)
                            dst_chain[k * size + R1] = chain[slot][k * size + T];
                        dst_chain[slot * size + R1] = static_cast<int32_t>(mask);
                    }
                    if (T == 0) break;
                    T = (T - 1) & comp;
                }
            }
        }
    }

    // Collapse over the number of used slots; trailing vehicles stay empty.
    py::array_t<double> g_out(size);
    py::array_t<int32_t> sets_out({p, size});
    double* g = g_out.mutable_data();
    int32_t* sets = sets_out.mutable_data();
    const double* tl = tail.data();
    for (py::ssize_t R = 0; R < size; ++R) {
        double best = kInf;
        py::ssize_t best_i = 0;
        for (py::ssize_t i = 0; i <= p; ++i) {
            const double v = f[i][R] + tl[p - i];
            if (v < best) {
                best = v;
                best_i = i;
            }
        }
        g[R] = best;
        for (py::ssize_t k = 1; k <= p; ++k)
            sets[(k - 1) * size + R] =
                (k <= best_i) ? chain[best_i][(k - 1) * size + R] : 0;
    }
    return py::make_tuple(g_out, sets_out);
}

}  // namespace

PYBIND11_MODULE(subset_dp_kernel, m) {
    m.doc() = "Subset-DP kernels for the Stage-2 fixed-fleet piece solver";
    m.def("theta_table", &theta_table, py::arg("coeffs"), py::arg("intercept_used"),
          py::arg("intercept_empty"), py::arg("feasible"));
    m.def("price_table", &price_table, py::arg("theta"), py::arg("item_coeff"),
          py::arg("nonempty_coeff"), py::arg("constant"), py::arg("top_k") = 8,
          py::arg("pair_masks") = py::array_t<int64_t>(0),
          py::arg("pair_coeff") = py::array_t<double>(0),
          py::arg("apart_masks") = py::array_t<int64_t>(0));
    m.def("terminal_single_layer", &terminal_single_layer, py::arg("g"),
          py::arg("theta"), py::arg("subset_reward"), py::arg("tail"));
    m.def("type_layer", &type_layer, py::arg("g"), py::arg("thetas"), py::arg("masks_desc"),
          py::arg("group_starts"), py::arg("tail"));
}
