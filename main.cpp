#include <array>
#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "graph.hpp"

using namespace std::chrono;

std::string bench_traverse(std::function<void()> traverse_fn) {
    auto start = high_resolution_clock::now();
    traverse_fn();
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(stop - start);
    return std::to_string(duration.count());
}

void full_bench(Graph& graph) {
    int num_test = 1;
    std::array<int, 6> num_threads{{1, 2, 4, 8, 16, 32}};

    std::vector<Graph::Node> visited(graph.size(), false);
    Graph::Node src = 0;

    omp_set_dynamic(0);

    std::cout << "Number of nodes: " << graph.size() << "\n\n";

    for (int i = 0; i < num_test; i++) {
        std::cout << "\tExecution " << i + 1 << std::endl;

        std::cout << "Sequential iterative DFS: "
                  << bench_traverse([&] { graph.dfs(src, visited); }) << "ms\n";

        std::fill(visited.begin(), visited.end(), false);

        std::cout << "Sequential recursive DFS: "
                  << bench_traverse([&]() { graph.rdfs(src, visited); }) << "ms\n";

        std::cout << "Sequential iterative BFS: "
                  << bench_traverse([&] { graph.dijkstra(src); }) << "ms\n";

        for (const auto n : num_threads) {
            std::fill(visited.begin(), visited.end(), false);

            std::cout << "Using " << n << " threads..." << std::endl;

            omp_set_num_threads(n);

            std::cout << "Parallel iterative DFS: "
                      << bench_traverse([&] { graph.p_dfs(src, visited); }) << "ms\n";

            std::fill(visited.begin(), visited.end(), false);

            std::cout << "Parallel recursive DFS: "
                      << bench_traverse([&] { graph.p_rdfs(src, visited); }) << "ms\n";

            std::cout << "Parallel iterative BFS: "
                      << bench_traverse([&] { graph.p_dijkstra(src); }) << "ms\n";
        }

        std::fill(visited.begin(), visited.end(), false);
        std::cout << std::endl;
    }
}

int main(int argc, const char** argv) {
    if (argc < 2) {
        std::cout << "Input file not specified.\n";
        return 1;
    }

    std::string file_path = argv[1];
    auto graph = import_graph(file_path);

    full_bench(graph);

    return 0;
}
