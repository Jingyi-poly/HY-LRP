// SPDX-License-Identifier: GPL-3.0-or-later
// Experimental adapter to the unmodified RouteOpt CVRP pricing implementation.
#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "cvrp_pricing_controller.hpp"

using namespace RouteOpt;
using namespace RouteOpt::Application::CVRP;

int main() {
    const auto started = std::chrono::steady_clock::now();
    auto *output = std::cout.rdbuf();
    std::ostringstream diagnostics;
    std::cout.rdbuf(diagnostics.rdbuf());
    try {
        int n, capacity, max_routes, ng_size;
        double seconds;
        int64_t beta;
        std::string protocol;
        if (!(std::cin >> protocol) || protocol != "ROUTEOPT_GRID_PRICING_2" ||
            !(std::cin >> n >> capacity >> beta >> seconds >> max_routes >> ng_size))
            throw std::runtime_error("invalid input header");
        const int dim = n + 1;
        const int max_vehicles = 1;
        if (n < 1 || n >= MAX_NUM_CUSTOMERS || capacity < 1 ||
            !(seconds > 0) || max_routes < 1 || ng_size < 0)
            throw std::runtime_error("invalid pricing dimensions or budget");
        std::vector<double> demands(dim, 0), duals(dim), zeros(dim, 0);
        for (int j = 1; j < dim; ++j) {
            int demand;
            if (!(std::cin >> demand) || demand <= 0)
                throw std::runtime_error("pricing resources must be positive integers");
            demands[j] = demand;
        }
        for (int j = 0; j < n; ++j) {
            int64_t price;
            if (!(std::cin >> price)) throw std::runtime_error("missing price");
            duals[j] = static_cast<double>(price);
        }
        duals[n] = static_cast<double>(beta);
        std::vector<std::vector<double>> costs(dim, std::vector<double>(dim));
        for (auto &row: costs) for (auto &value: row) {
            int64_t cost;
            if (!(std::cin >> cost)) throw std::runtime_error("missing cost");
            value = static_cast<double>(cost);
        }
        Rank1Cuts::Rank1CutsDataShared shared(dim);
        Rank1Cuts::RCGetter::Rank1RCController rank1(shared);
        CVRP_Pricing pricing(dim, max_vehicles, capacity, demands, 0, zeros, zeros, zeros, costs, rank1);
        if (ng_size == 0 || ng_size >= n) {
            routeOptLong all_customers;
            for (int j = 1; j < dim; ++j) all_customers.set(j);
            for (auto &memory: pricing.refNG()) memory = all_customers;
        } else if (ng_size != InitialNGSize) {
            for (int i = 0; i < dim; ++i) {
                std::vector<std::pair<double, int>> neighbors;
                for (int j = 1; j < dim; ++j) neighbors.emplace_back(costs[i][j], j);
                std::stable_sort(neighbors.begin(), neighbors.end());
                auto &memory = pricing.refNG()[i];
                memory.reset();
                for (int k = 0; k < ng_size; ++k) memory.set(neighbors[k].second);
            }
        }
        pricing.refNumColGeneratedUB() = max_routes;
        pricing.initLabelingMemory();
        Bucket **forward = nullptr, **backward = nullptr;
        std::vector<std::vector<std::vector<int>>> forward_order, backward_order;
        pricing.updatePtr(forward, backward, &forward_order, &backward_order);
        int forward_arcs = 0, backward_arcs = 0;
        pricing.initializeBucketGraphForNode<true>(forward, backward, forward_arcs, backward_arcs);
        pricing.initializeOnceB4WholePricing();
        pricing.priceConstraints({}, {}, {}, duals);
        pricing.generateColumnsByExact<true>(seconds);
        const bool complete = pricing.getIfCompleteCG();
        const double minimum = pricing.getSmallestRC();
        const auto routes = pricing.getNewCols();
        for (int j = 0; j < dim; ++j) delete[] forward[j];
        delete[] forward;
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::cout.rdbuf(output);
        std::cout << std::setprecision(17) << "{\"pricing_complete\":"
                  << (complete ? "true" : "false") << ",\"min_rc\":";
        if (complete) std::cout << minimum;
        else std::cout << "null";
        std::cout << ",\"kernel_seconds\":" << elapsed << ",\"routes\":[";
        for (size_t r = 0; r < routes.size(); ++r) {
            if (r) std::cout << ',';
            std::cout << '[';
            for (size_t j = 0; j < routes[r].col_seq.size(); ++j) {
                if (j) std::cout << ',';
                std::cout << routes[r].col_seq[j];
            }
            std::cout << ']';
        }
        std::cout << "]}" << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cout.rdbuf(output);
        std::cerr << error.what() << '\n';
        std::cout << "{\"pricing_complete\":false,\"min_rc\":null,\"routes\":[]}" << std::endl;
        return 2;
    }
}
