// This code is a modified version of the code found in Antoine Savine's book "Modern Computational Finance - AAD and Parallel Simulations".
#include "main.hpp"
#include <iostream>
#include <chrono>

void price_cliquet() {
    ThreadPool* pool = ThreadPool::get_pool();
    pool->start();
    Sobol rng;
    const size_t n_path = 1'000'000;
    const double T = 1.0;             // total maturity (1 year)
    const size_t n_obs = 12;          // monthly observations
    const double dt = T / n_obs;
    Number spot = Number(100.0), volatility = Number(0.20);

    std::vector<Time> obs_dates;
    obs_dates.reserve(n_obs + 1);
    for (size_t i = 0; i <= n_obs; ++i)
        obs_dates.push_back(i * dt);

    const double local_floor = -0.02;       // -2% per period floor
    const double local_cap   =  0.05;       // +5% per period cap
    const double global_floor = 0.0;        // optional global floor
    const double global_cap   = 0.20;       // optional global cap
    const Time settlement_date = T;

    auto model = BlackScholes<Number>(spot, volatility);
    auto product = Cliquet<Number>(
        obs_dates, 
        settlement_date,
        local_floor, 
        local_cap, 
        global_floor, 
        global_cap
    );

    auto result = mc_parallel_simul_aad(product, model, rng, n_path);

    std::cout << "==== Cliquet Option Pricing ====" << std::endl;
    std::cout << "Delta : " << result.risks[0] << std::endl;
    std::cout << "Vega  : " << result.risks[1] << std::endl;

    pool->stop();
}

void price_european() {
    ThreadPool* pool = ThreadPool::get_pool();
    pool->start();
    Sobol rng;
    const size_t n_path = 1'000'000;
    const double strike = 100.0;
    const double exercise_date = 1.0;
    Number spot = Number(100.0), volatility = Number(0.20);

    auto model = BlackScholes<Number>(spot, volatility);
    auto product = European<Number>(strike, exercise_date);

    auto result = mc_parallel_simul_aad(product, model, rng, n_path);

    std::cout << "==== European Option Pricing ====" << std::endl;
    std::cout << "Delta : " << result.risks[0] << std::endl;
    std::cout << "Vega  : " << result.risks[1] << std::endl;

    pool->stop();
}

int main(){
    price_cliquet();
    price_european();
    return 0;
}