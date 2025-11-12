// This code is a modified version of the code found in Antoine Savine's book "Modern Computational Finance - AAD and Parallel Simulations".
#pragma once

#include <algorithm>
#include <map>
#include "simul.hpp"
#include <sstream>

#define ONE_HOUR 0.000114469
#define ONE_DAY 0.003773585

using std::max;

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
