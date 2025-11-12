// This code is a modified version of the code found in Antoine Savine's book "Modern Computational Finance - AAD and Parallel Simulations".
#include "main.hpp"
#include <iostream>

int main(){
    ThreadPool* pool = ThreadPool::get_pool();
    pool->start();
    Sobol rng;
    size_t n_path = 10'000'000;
    Number spot = Number(100.0), volatility = Number(0.20);
    double strike = 100.0, exercise_date = 1.0;
    auto model = BlackScholes<Number>(spot, volatility);

    auto product = European<Number>(strike, exercise_date);

    auto result = mc_parallel_simul_aad(product, model, rng, n_path);
    std::cout << "Delta: " << result.risks[0] << std::endl;
    std::cout << "Vega: " << result.risks[1] << std::endl;
    return 0;
}