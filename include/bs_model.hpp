// This code is a modified version of the code found in Antoine Savine's book "Modern Computational Finance - AAD and Parallel Simulations".
#pragma once
#include <vector>
#include <string>
#include "simul.hpp"

using std::max, std::sqrt, std::exp;

template <class T>
class BlackScholes : public Model<T> {
    T spot_;
    T rate_;
    T dividend_;
    T volatility_;
    const bool spot_measure_;
    std::vector<Time> timeline_;
    bool today_on_timeline_;
    const std::vector<SampleDef>* definition_line_;
    std::vector<T> stds_;
    std::vector<T> drifts_;
    std::vector<T> numeraires_;
    std::vector<std::vector<T>> discounts_;
    std::vector<std::vector<T>> forward_factors_;
    std::vector<std::vector<T>> libors_;
    std::vector<T*> parameters_;
    std::vector<std::string> parameter_labels_;

    public:

        template <class U>
        BlackScholes(
            const U spot,
            const U vol,
            const bool spot_measure = false,
            const U rate = U(0.0),
            const U div = U(0.0)
        ) : 
        spot_(spot), 
        volatility_(vol), 
        rate_(rate), 
        dividend_(div), 
        spot_measure_(spot_measure), 
        parameters_(4), 
        parameter_labels_(4) 
        {
            parameter_labels_[0] = "spot";
            parameter_labels_[1] = "vol";
            parameter_labels_[2] = "rate";
            parameter_labels_[3] = "div";

            set_param_pointers();
        }

    private:

        void set_param_pointers() {
            parameters_[0] = &spot_;
            parameters_[1] = &volatility_;
            parameters_[2] = &rate_;
            parameters_[3] = &dividend_;
        }

    public:

        T spot() const {
            return spot_;
        }

        const T vol() const {
            return volatility_;
        }

        const T rate() const {
            return rate_;
        }

        const T div() const {
            return dividend_;
        }

        const std::vector<T*>& parameters() override {
            return parameters_;
        }
        const std::vector<std::string>& parameter_labels() const override {
            return parameter_labels_;
        }

        std::unique_ptr<Model<T>> clone() const override {
            auto clone = std::make_unique<BlackScholes<T>>(*this);
            clone->set_param_pointers();
            return clone;
        }

        void allocate(const std::vector<Time>& product_timeline, const std::vector<SampleDef>& definition_line) override {
            timeline_.clear();
            timeline_.push_back(system_time);
            for (const auto& time : product_timeline) {
                if (time > system_time) timeline_.push_back(time);
            }

            today_on_timeline_ = (product_timeline[0] == system_time);

            definition_line_ = &definition_line;

            stds_.resize(timeline_.size() - 1);
            drifts_.resize(timeline_.size() - 1);

            const size_t n = product_timeline.size();
            numeraires_.resize(n);
            
            discounts_.resize(n);
            for (size_t j = 0; j < n; ++j) {
                discounts_[j].resize(definition_line[j].discount_maturities.size());
            }

            forward_factors_.resize(n);
            for (size_t j = 0; j < n; ++j) {
                forward_factors_[j].resize(definition_line[j].forward_maturities.size());
            }

            libors_.resize(n);
            for (size_t j = 0; j < n; ++j) {
                libors_[j].resize(definition_line[j].libor_definitions.size());
            }
        }

        void init(const std::vector<Time>& product_timeline, const std::vector<SampleDef>& definition_line) override {     
            const T mu = rate_ - dividend_;
            const size_t n = timeline_.size() - 1;

            for (size_t i = 0; i < n; ++i) {
                const double dt = timeline_[i + 1] - timeline_[i];
                stds_[i] = volatility_ * sqrt(dt);
                
                if (spot_measure_) {
                    drifts_[i] = (mu + 0.5*volatility_*volatility_)*dt;
                } else {
                    drifts_[i] = (mu - 0.5*volatility_*volatility_)*dt;
                }
            }

            const size_t m = product_timeline.size();

            for (size_t i = 0; i < m; ++i) {
                if (definition_line[i].numeraire) {
                    if (spot_measure_) {
                        numeraires_[i] = exp(dividend_ * product_timeline[i]) / spot_;
                    } else {
                        numeraires_[i] = exp(rate_ * product_timeline[i]);
                    }
                }

                const size_t pDF = definition_line[i].discount_maturities.size();
                for (size_t j = 0; j < pDF; ++j) {
                    discounts_[i][j] =
                        exp(-rate_ * (definition_line[i].discount_maturities[j] - product_timeline[i]));
                }

                const size_t pFF = definition_line[i].forward_maturities.front().size();
                for (size_t j = 0; j < pFF; ++j) {
                    forward_factors_[i][j] =
                        exp(mu * (definition_line[i].forward_maturities.front()[j] - product_timeline[i]));
                }

                const size_t pL = definition_line[i].libor_definitions.size();
                for (size_t j = 0; j < pL; ++j) {
                    const double dt
                        = definition_line[i].libor_definitions[j].end - definition_line[i].libor_definitions[j].start;
                    libors_[i][j] = (exp(rate_*dt) - 1.0) / dt;
                }
            }
        }

        size_t sim_dim() const override {
            return timeline_.size() - 1;
        }

    private:

        inline void fill_scenario(const size_t idx, const T& spot, Sample<T>& scen, const SampleDef& def) const {
            if (def.numeraire) {
                scen.numeraire = numeraires_[idx];
                if (spot_measure_) scen.numeraire *= spot;
            }
            
            std::transform(forward_factors_[idx].begin(), forward_factors_[idx].end(), 
                scen.forwards.front().begin(), 
                [&spot](const T& ff) {
                    return spot * ff;
                }
            );

            std::copy(discounts_[idx].begin(), discounts_[idx].end(), scen.discounts.begin());
            std::copy(libors_[idx].begin(), libors_[idx].end(), scen.libors.begin());
        }

    public:

        void generate_path(const std::vector<double>& gaussian_noise, Scenario<T>& path) const override {
            T spot = spot_;
            size_t idx = 0;
            if (today_on_timeline_) {
                fill_scenario(idx, spot, path[idx], (*definition_line_)[idx]);
                ++idx;
            }

            const size_t n = timeline_.size() - 1;
            for (size_t i = 0; i < n; ++i) {
                spot = spot * exp(drifts_[i] + stds_[i] * gaussian_noise[i]);
                fill_scenario(idx, spot, path[idx], (*definition_line_)[idx]);
                ++idx;
            }
        }
};