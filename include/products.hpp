// This code is a modified version of the code found in Antoine Savine's book "Modern Computational Finance - AAD and Parallel Simulations".
#pragma once

#include <algorithm>
#include <map>
#include "simul.hpp"
#include <sstream>

#define ONE_HOUR 0.000114469
#define ONE_DAY 0.003773585

using std::max;
using std::min;

template <class T>
class European : public Product<T> {
    double strike_;
    Time exercise_date_;
    Time settlement_date_;

    std::vector<Time> timeline_;
    std::vector<SampleDef> definition_line_;

    std::vector<std::string> labels_;

    public:
        
        European(
            const double strike, 
            const Time exercise_date,
            const Time settlement_date) :
            strike_(strike),
            exercise_date_(exercise_date),
            settlement_date_(settlement_date),
            labels_(1) {
            timeline_.push_back(exercise_date);

            definition_line_.resize(1);
            SampleDef& sample_definition = definition_line_.front();

            sample_definition.numeraire = true;
            sample_definition.forward_maturities.push_back({settlement_date});
            sample_definition.discount_maturities.push_back(settlement_date);

            //  Identify the product
            std::ostringstream ost;
            ost.precision(2);
            ost << std::fixed;
            if (settlement_date == exercise_date) {
                ost << "call " << strike_ << " " 
                    << exercise_date;
            } else {
                ost << "call " << strike_ << " " 
                    << exercise_date << " " << settlement_date;
            }
            labels_[0] = ost.str();
        }

        European(const double strike, const Time exercise_date) : European(strike, exercise_date, exercise_date) {}

        std::unique_ptr<Product<T>> clone() const override {return std::make_unique<European<T>>(*this);}
        const std::vector<Time>& timeline() const override {return timeline_;}
        const std::vector<SampleDef>& definition_line() const override {return definition_line_;}
        const std::vector<std::string>& payoff_labels() const override {return labels_;}

        void payoffs(const Scenario<T>& path, std::vector<T>& payoffs) const override {
            const auto& sample = path.front();
            const auto spot = sample.forwards.front().front();
            payoffs.front() = max(spot - strike_, 0.0) * sample.discounts.front() / sample.numeraire; 
        }
};

template <class T>
class Cliquet : public Product<T> {
    double local_floor_;
    double local_cap_;
    double global_floor_;
    double global_cap_;
    std::vector<Time> observation_dates_;
    Time settlement_date_;

    std::vector<SampleDef> definition_line_;
    std::vector<std::string> labels_;

public:
    Cliquet(
        const std::vector<Time>& observation_dates,
        const Time settlement_date,
        double local_floor,
        double local_cap,
        double global_floor = -1.0,  // optional
        double global_cap = 1.0      // optional
    ) : 
        local_floor_(local_floor),
        local_cap_(local_cap),
        global_floor_(global_floor),
        global_cap_(global_cap),
        observation_dates_(observation_dates),
        settlement_date_(settlement_date),
        labels_(1) {
        definition_line_.resize(observation_dates_.size());
        for (size_t i = 0; i < observation_dates_.size(); ++i) {
            SampleDef& sample = definition_line_[i];
            sample.numeraire = true;
            sample.forward_maturities.push_back({observation_dates_[i]});
            sample.discount_maturities.push_back(settlement_date_);
        }

        std::ostringstream ost;
        ost << "cliquet [" << observation_dates_.front()
            << "..." << observation_dates_.back() << "]";
        labels_[0] = ost.str();
    }

    std::unique_ptr<Product<T>> clone() const override {return std::make_unique<Cliquet<T>>(*this);}
    const std::vector<Time>& timeline() const override {return observation_dates_;}
    const std::vector<SampleDef>& definition_line() const override {return definition_line_;}
    const std::vector<std::string>& payoff_labels() const override {return labels_;}

    void payoffs(const Scenario<T>& path, std::vector<T>& payoffs) const override {
        T sum_local_returns = T(0.0);
        for (size_t i = 1; i < path.size(); ++i) {
            const auto& prev_sample = path[i - 1];
            const auto& curr_sample = path[i];

            const T spot_prev = prev_sample.forwards.front().front();
            const T spot_curr = curr_sample.forwards.front().front();

            T local_return = (spot_curr / spot_prev) - 1.0;
            local_return = max(local_floor_, min(local_cap_, local_return));

            sum_local_returns += local_return;
        }

        // Apply global cap/floor
        T global_return = max(global_floor_, min(global_cap_, sum_local_returns));

        // Determine final payoff
        const T S0 = path.front().forwards.front().front();
        global_return *= S0;

        // Discount payoff
        const auto& last_sample = path.back();
        payoffs.front() = global_return * last_sample.discounts.front() / last_sample.numeraire;
    }
};