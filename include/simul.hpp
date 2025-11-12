// This code is a modified version of the code found in Antoine Savine's book "Modern Computational Finance - AAD and Parallel Simulations".
#pragma once

#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <threadpool.hpp>
#include <string>
#include "number.hpp"

using Time = double;
using std::min;
extern Time system_time;

template<class It1, class It2>
inline void convert_collection(It1 srcBegin, It1 srcEnd, It2 destBegin) {
    using destType = std::remove_reference_t<decltype(*destBegin)>;
    std::transform(srcBegin, srcEnd, destBegin, [](const auto& source) {return destType(source);});
}

struct SampleDef {
    bool numeraire = true;

    struct RateDef {
        Time start;
        Time end;
        std::string curve;

        RateDef(const Time s, const Time e, const std::string& c) : start(s), end(e), curve(c) {};
    };

    std::vector<Time> discount_maturities;
    std::vector<RateDef> libor_definitions;
    std::vector<std::vector<Time>> forward_maturities;
};

template <class T>
struct Sample {
    T numeraire;
    std::vector<T> discounts;
    std::vector<T> libors;
    std::vector<std::vector<T>> forwards;

    void allocate(const SampleDef& data) {
        discounts.resize(data.discount_maturities.size());
        libors.resize(data.libor_definitions.size());
        forwards.resize(data.forward_maturities.size());
        for (size_t a = 0; a < forwards.size(); ++a) forwards[a].resize(data.forward_maturities[a].size());
    }

    void initialise() {
        numeraire = T(1.0);
        std::fill(discounts.begin(), discounts.end(), T(1.0));
        std::fill(libors.begin(), libors.end(), T(0.0));
		for (auto& forward: forwards) std::fill(forward.begin(), forward.end(), T(100.0));
    }
};

template <class T>
using Scenario = std::vector<Sample<T>>;
template <class T>
inline void allocate_path(const std::vector<SampleDef>& definition_line, Scenario<T>& path) {
    path.resize(definition_line.size());
    for (size_t i = 0; i < definition_line.size(); ++i) {
        path[i].allocate(definition_line[i]);
    }
}

template <class T>
inline void initialise_path(Scenario<T>& path) {
    for (auto& scenario : path) scenario.initialise();
}


template <class T>
class Product {
    inline static const std::vector<std::string> default_asset_names = { "spot" };

    public:
        virtual const std::vector<Time>& timeline() const = 0;
        virtual const std::vector<SampleDef>& definition_line() const = 0;
        virtual const size_t num_assets() const {return 1;}
        virtual const std::vector<std::string>& asset_names() const {return default_asset_names;}
        virtual const std::vector<std::string>& payoff_labels() const = 0;
        virtual void payoffs(const Scenario<T>& path, std::vector<T>& payoffs) const = 0;
        virtual std::unique_ptr<Product<T>> clone() const = 0;
        virtual ~Product() {}
};

template <class T>
class Model {
    inline static const std::vector<std::string> default_asset_names = { "spot" };

    public:
        virtual const size_t num_assets() const {return 1;}
        virtual const std::vector<std::string>& asset_names() const {return default_asset_names;}
        virtual void allocate(const std::vector<Time>& product_timeline, const std::vector<SampleDef>& product_def_line) = 0;
        virtual void init(const std::vector<Time>& product_timeline, const std::vector<SampleDef>& product_def_line) = 0;
        virtual size_t sim_dim() const = 0;
        virtual void generate_path(const std::vector<double>& gaussian_noise, Scenario<T>& path) const = 0;
        virtual std::unique_ptr<Model<T>> clone() const = 0;
        virtual ~Model() {}
        virtual const std::vector<T*>& parameters() = 0;
        virtual const std::vector<std::string>& parameter_labels() const = 0;
        size_t num_params() const {return const_cast<Model*>(this)->parameters().size();}
        void put_parameters_on_tape(){put_parameters_on_tapeT<T>();}

    private:
        template<class U> 
        void put_parameters_on_tapeT() {}
        template <>
        void put_parameters_on_tapeT<Number>(){for (Number* param : parameters()) param->put_on_tape();}
};

class RNG {
public:
    virtual void init(const size_t sim_dim) = 0;
	virtual void next_uniform(std::vector<double>& uniform_noise) = 0;
	virtual void next_gaussian(std::vector<double>& gaussian_noise) = 0;
    virtual std::unique_ptr<RNG> clone() const = 0;
    virtual ~RNG() {}
    virtual void skip_to(const unsigned b) = 0;
};

template <class T>
inline bool check_compatiblity(const Product<T>& product, const Model<T>& model) {return product.asset_names() == model.asset_names();}

inline std::vector<std::vector<double>> mc_simul(
    const Product<double>& product,
    const Model<double>& model,
    const RNG& rng,			            
    const size_t n_path) {
    if (!check_compatiblity(product, model)) throw std::runtime_error("Model and product are not compatible");

    auto model_clone = model.clone();
    auto rng_clone = rng.clone();

    const size_t n_payoffs = product.payoff_labels().size();
    std::vector<std::vector<double>> results(n_path, std::vector<double>(n_payoffs));
    model_clone->allocate(product.timeline(), product.definition_line());
    model_clone->init(product.timeline(), product.definition_line());              

    rng_clone->init(model_clone->sim_dim());                        
    
    std::vector<double> gaussian_noise(model_clone->sim_dim());           
    Scenario<double> path;
    allocate_path(product.definition_line(), path);
    initialise_path(path);

    for (size_t i = 0; i < n_path; i++) {
        rng_clone->next_gaussian(gaussian_noise);
        model_clone->generate_path(gaussian_noise, path);
        product.payoffs(path, results[i]);
    }
    return results;
}


#define BATCHSIZE size_t{64}
inline std::vector<std::vector<double>> mc_parallel_simul(
    const Product<double>& product,
    const Model<double>& model,
    const RNG& rng,
    const size_t n_path) {
    if (!check_compatiblity(product, model)) throw std::runtime_error("Model and product are not compatible");

    auto model_clone = model.clone();

    const size_t n_payoffs = product.payoff_labels().size();
    std::vector<std::vector<double>> results(n_path, std::vector<double>(n_payoffs));

    model_clone->allocate(product.timeline(), product.definition_line());
    model_clone->init(product.timeline(), product.definition_line());

    ThreadPool *pool = ThreadPool::get_pool();
    const size_t n_thread = pool->n_threads();
    std::vector<std::vector<double>> gaussian_noises(n_thread + 1);
    std::vector<Scenario<double>> paths(n_thread+1);
    for (auto& vec : gaussian_noises) vec.resize(model_clone->sim_dim());
    for (auto& path : paths) {
        allocate_path(product.definition_line(), path);
        initialise_path(path);
    }
    
    std::vector<std::unique_ptr<RNG>> rngs(n_thread + 1);
    for (auto& random : rngs) {
        random = rng.clone();
        random->init(model_clone->sim_dim());
    }

    std::vector<TaskHandle> futures;
    futures.reserve(n_path / BATCHSIZE + 1); 

    size_t first_path = 0;
    size_t paths_left = n_path;
    while (paths_left > 0) {
        size_t paths_in_task = std::min<size_t>(paths_left, BATCHSIZE);

        futures.push_back( pool->spawn_task( [&, first_path, paths_in_task]() {
            const size_t thread_num = pool->thread_num();
            std::vector<double>& gaussian_noise = gaussian_noises[thread_num];
            Scenario<double>& path = paths[thread_num];

            auto& random = rngs[thread_num];
            random->skip_to(first_path);

            for (size_t i = 0; i < paths_in_task; i++) {
                random->next_gaussian(gaussian_noise);
                model_clone->generate_path(gaussian_noise, path);       
                product.payoffs(path, results[first_path + i]);
            }
            return true;
        }));

        paths_left -= paths_in_task;
        first_path += paths_in_task;
    }
    
    for (auto& future : futures) pool->active_wait(future);
    
    return results;
}

struct AADSimulResults
{
    AADSimulResults(const size_t n_path, const size_t n_payoffs, const size_t n_param) 
    : payoffs(n_path, std::vector<double>(n_payoffs)), aggregated(n_path), risks(n_param) {}

    std::vector<std::vector<double>> payoffs;
    std::vector<double> aggregated;
    std::vector<double> risks;
};

const auto default_aggregator = [](const std::vector<Number>& v) {return v[0];};

template<class F = decltype(default_aggregator)>
inline AADSimulResults
mc_simul_aad(
    const Product<Number>& product,
    const Model<Number>& model,
    const RNG& rng,
    const size_t n_path,
    const F& agg_fun = default_aggregator) {
    if (!check_compatiblity(product, model)) throw std::runtime_error("Model and product are not compatible");

    auto model_clone = model.clone();
    auto rng_clone = rng.clone();

	Scenario<Number> path;
    allocate_path(product.definition_line(), path);
	model_clone->allocate(product.timeline(), product.definition_line());

    const size_t n_payoffs = product.payoff_labels().size();
    const std::vector<Number*>& params = model_clone->parameters();
    const size_t n_param = params.size();

    Tape& tape = *Number::tape;
    tape.clear();
    model_clone->put_parameters_on_tape();
    model_clone->init(product.timeline(), product.definition_line());
    initialise_path(path);
    tape.mark();

    rng_clone->init(model_clone->sim_dim());                         
    
    std::vector<Number> payoffs(n_payoffs);
    std::vector<double> gaussian_noise(model_clone->sim_dim());            
    AADSimulResults results(n_path, n_payoffs, n_param);

    for (size_t i = 0; i < n_path; i++) {
        tape.rewind_to_mark();
        rng_clone->next_gaussian(gaussian_noise);
        model_clone->generate_path(gaussian_noise, path);     
        product.payoffs(path, payoffs);
        Number result = agg_fun(payoffs);
        result.propagate_to_mark();
        results.aggregated[i] = double(result);
        convert_collection(
            payoffs.begin(), 
            payoffs.end(), 
            results.payoffs[i].begin()
        );
    }

    Number::propagate_mark_to_start();

    std::transform(
        params.begin(),
        params.end(),
        results.risks.begin(),
        [n_path](const Number* p) {return p->adjoint() / n_path;}
    );
    tape.clear();
    return results;
}

inline void init_model_parallel(
    const Product<Number>& product,
    Model<Number>& model_clone,
    Scenario<Number>& path) {
    Tape& tape = *Number::tape;
    tape.rewind();
    model_clone.put_parameters_on_tape();
    model_clone.init(product.timeline(), product.definition_line());
    initialise_path(path);
    tape.mark();
}

template<class F = decltype(default_aggregator)>
inline AADSimulResults
mc_parallel_simul_aad(
    const Product<Number>& product,
    const Model<Number>& model,
    const RNG& rng,
    const size_t n_path,
    const F& agg_fun = default_aggregator
) {
    if (!check_compatiblity(product, model)) throw std::runtime_error("Model and product are not compatible");

    const size_t n_payoffs = product.payoff_labels().size();
    const size_t n_param = model.num_params();

    AADSimulResults results(n_path, n_payoffs, n_param);

	Number::tape->clear();

    ThreadPool *pool = ThreadPool::get_pool();
    const size_t n_threads = pool->n_threads();

    std::vector<std::unique_ptr<Model<Number>>> models(n_threads + 1);
    for (auto& m : models) {
        m = model.clone();
        m->allocate(product.timeline(), product.definition_line());
    }

    std::vector<Scenario<Number>> paths(n_threads + 1);
    for (auto& path : paths) {
        allocate_path(product.definition_line(), path);
    }

    std::vector<std::vector<Number>> payoffs(n_threads + 1, std::vector<Number>(n_payoffs));

    std::vector<Tape> tapes(n_threads);

    std::vector<int> model_init(n_threads + 1, false);

    init_model_parallel(product, *models[0], paths[0]);

    model_init[0] = true;

    std::vector<std::unique_ptr<RNG>> rngs(n_threads + 1);
    for (auto& random : rngs) {
        random = rng.clone();
        random->init(models[0]->sim_dim());
    }

    std::vector<std::vector<double>> gaussian_vectors(n_threads + 1, std::vector<double>(models[0]->sim_dim()));

    std::vector<TaskHandle> futures;
    futures.reserve(n_path / BATCHSIZE + 1);

    size_t first_path = 0;
    size_t paths_left = n_path;
    while (paths_left > 0) {
        size_t paths_in_task = min<size_t>(paths_left, BATCHSIZE);
        futures.push_back(pool->spawn_task([&, first_path, paths_in_task]() {
            const size_t thread_num = pool->thread_num();
            if (thread_num > 0) Number::tape = &tapes[thread_num - 1];
            if (!model_init[thread_num]) {
                init_model_parallel(product, *models[thread_num], paths[thread_num]);
                model_init[thread_num] = true;
            }

            auto& random = rngs[thread_num];
            random->skip_to(first_path);

            for (size_t i = 0; i < paths_in_task; i++) {
                Number::tape->rewind_to_mark();
                random->next_gaussian(gaussian_vectors[thread_num]);
                models[thread_num]->generate_path(gaussian_vectors[thread_num], paths[thread_num]);
                product.payoffs(paths[thread_num], payoffs[thread_num]);
                Number result = agg_fun(payoffs[thread_num]);
                result.propagate_to_mark();
                results.aggregated[first_path + i] = double(result);
                convert_collection(
                    payoffs[thread_num].begin(), 
                    payoffs[thread_num].end(),
                    results.payoffs[first_path + i].begin()
                );
            }

            return true;
        }));

        paths_left -= paths_in_task;
        first_path += paths_in_task;
    }
    for (auto& future : futures) pool->active_wait(future);
    Number::propagate_mark_to_start();
    Tape* main_thread_ptr = Number::tape;
    for (size_t i = 0; i < n_threads; ++i) {
        if (model_init[i + 1]) {
            Number::tape = &tapes[i];
            Number::propagate_mark_to_start();
        }
    }
    Number::tape = main_thread_ptr;
    for (size_t j = 0; j < n_param; ++j) {
        results.risks[j] = 0.0;
        for (size_t i = 0; i < models.size(); ++i) {
            if (model_init[i]) results.risks[j] += models[i]->parameters()[j]->adjoint();
        }
        results.risks[j] /= n_path;
    }
    Number::tape->clear();
    return results;
    }