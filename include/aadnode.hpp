// This code is a modified version of the code found in Antoine Savine's book "Modern Computational Finance - AAD and Parallel Simulations".
#pragma once

class Node {
    friend class Tape;
    friend class Number;

    const size_t n;
    double adjoint_ = 0;
    double* derivatives_;
    double** arg_adjoint_ptr_;

    public:
        Node(const size_t N = 0) : n(N) {};

        double& adjoint(){
            return adjoint_;
        };

        void propagate(){
            if (!n || !adjoint_) return;
            for (size_t i = 0; i < n; ++i){
                *(arg_adjoint_ptr_[i]) += derivatives_[i] * adjoint_;
            }
        }
};