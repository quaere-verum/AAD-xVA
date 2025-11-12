// This code is a modified version of the code found in Antoine Savine's book "Modern Computational Finance - AAD and Parallel Simulations".
#pragma once

#include "simul.hpp"
#include "gaussian.hpp"

#define ONEOVER2POW32 2.3283064365387E-10

const unsigned * const * getjkDir();

class Sobol : public RNG {
    size_t dim_;
    std::vector<unsigned> state_;  
    unsigned index_;
    const unsigned * const * jkDir;
    
    public:

        std::unique_ptr<RNG> clone() const override {return std::make_unique<Sobol>(*this);}
        void init(const size_t sim_dim) override {
            jkDir = getjkDir();
            dim_ = sim_dim;
            state_.resize(dim_);
            reset();
        }

        void reset() {
            std::memset(state_.data(), 0, dim_ * sizeof(unsigned));
            index_ = 0;
        }
        
        void next() {
            unsigned n = index_, j = 0;
            while (n & 1) {
                n >>= 1;
                ++j;
            }
            const unsigned* dirNums = jkDir[j];
            for (int i = 0; i<dim_; ++i) {
                state_[i] ^= dirNums[i];
            }
            ++index_;
        }

        void next_uniform(std::vector<double>& uniform_vector) override {
            next();
            transform(
                state_.begin(), 
                state_.end(), 
                uniform_vector.begin(),
                [](const unsigned long i) {return ONEOVER2POW32 * i; }
            );
        }

        void next_gaussian(std::vector<double>& gaussian_vector) override {
            next();
            transform(
                state_.begin(), 
                state_.end(), 
                gaussian_vector.begin(),
                [](const unsigned long i) {return inverse_normal_cdf(ONEOVER2POW32 * i);});
        }

        void skip_to(const unsigned b) override {
            if (!b) return;
            reset();

            unsigned im = b;
            unsigned two_i = 1, two_i_plus_one = 2;

            unsigned i = 0;
            while (two_i <= im) {
                if (((im + two_i) / two_i_plus_one) & 1) {
                    for (unsigned k = 0; k<dim_; ++k) {
                        state_[k] ^= jkDir[i][k];
                    }
                }

                two_i <<= 1;
                two_i_plus_one <<= 1;
                ++i;
            }
            index_ = unsigned(b);
        }
};