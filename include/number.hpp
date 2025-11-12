// This code is a modified version of the code found in Antoine Savine's book "Modern Computational Finance - AAD and Parallel Simulations".
#pragma once
#include "aadnode.hpp"
#include "tape.hpp"
#include <cmath>

class Number {
    double value_;
    Node* node_;

    public:
        Number() {};

        explicit Number(const double val) : value_(val) {create_node<0>();}

        static thread_local Tape* tape;

        double& value() {
            return value_; 
        }

        double value() const {
            return value_;
        }

        double& adjoint() {
            return node_->adjoint_;
        }

        double adjoint() const {
            return node_->adjoint_;
        }

        void put_on_tape() {
            create_node<0>();
        }

        void reset_adjoints() {
            tape->reset_adjoints();
        }

        static void propagate_adjoints(Tape::Iterator propagate_from, Tape::Iterator propagate_to) {
            auto iter = propagate_from;
            while (iter != propagate_to) {
                iter->propagate();
                --iter;
            }
            iter->propagate();
        }

        void propagate_adjoints(Tape::Iterator propagate_to) {
            adjoint() = 1.0;
            auto propagate_from = tape->find(node_);
            propagate_adjoints(propagate_from, propagate_to);
        }

        void propagate_to_start() {
            propagate_adjoints(tape->begin());
        }

        void propagate_to_mark() {
            propagate_adjoints(tape->mark_iter());
        }

        static void propagate_mark_to_start() {
            propagate_adjoints(std::prev(tape->mark_iter()), tape->begin());
        }

        inline friend Number operator*(const Number& lhs, const Number& rhs) {
            const double v = lhs.value() * rhs.value();
            Number res(lhs.node(), rhs.node(), v);
            res.left_derivative() = rhs.value();
            res.right_derivative() = lhs.value();
            return res;
        }

        inline friend Number operator*(const Number& lhs, const double& rhs) {
            const double v = lhs.value() * rhs;
            Number res(lhs.node(), v);
            res.derivative() = rhs;
            return res;
        }

        inline friend Number operator*(const double& lhs, const Number& rhs) {
            return rhs * lhs;
        }

        inline friend Number operator+(const Number& lhs, const Number& rhs) {
            const double v = lhs.value() + rhs.value();
            Number res(lhs.node(), rhs.node(), v);
            res.left_derivative() = 1.0;
            res.right_derivative() = 1.0;
            return res;
        }

        inline friend Number operator+(const Number& lhs, const double& rhs) {
            const double v = lhs.value() + rhs;
            Number res(lhs.node(), v);
            res.derivative() = 1.0;
            return res;
        }

        inline friend Number operator+(const double& lhs, const Number& rhs) {
            return rhs + lhs;
        }

        inline friend Number operator-(const Number& lhs, const Number& rhs) {
            const double v = lhs.value() - rhs.value();
            Number res(lhs.node(), rhs.node(), v);
            res.left_derivative() = 1.0;
            res.right_derivative() = -1.0;
            return res;
        }

        inline friend Number operator-(const Number& lhs, const double& rhs) {
            const double v = lhs.value() - rhs;
            Number res(lhs.node(), v);
            res.derivative() = 1.0;
            return res;
        }

        inline friend Number operator-(const double& lhs, const Number& rhs) {
            const double v = lhs - rhs.value();
            Number res(rhs.node(), v);
            res.derivative() = -1.0;
            return res;
        }

        inline friend Number operator/(const Number& lhs, const Number& rhs) {
            const double v = lhs.value() / rhs.value();
            Number res(lhs.node(), rhs.node(), v);
            res.left_derivative() = 1.0 / rhs.value();
            res.right_derivative() = -v / rhs.value();
            return res;
        }

        inline friend Number operator/(const Number& lhs, const double& rhs) {
            const double v = lhs.value() / rhs;
            Number res(lhs.node(), v);
            res.derivative() = 1.0 / rhs;
            return res;
        }

        inline friend Number operator/(const double& lhs, const Number& rhs) {
            const double v = lhs / rhs.value();
            Number res(rhs.node(), v);
            res.derivative() = -v / rhs.value();
            return res;
        }

        inline friend Number pow(const Number& lhs, const Number& rhs) {
            const double base = lhs.value();
            const double exponent = rhs.value();
            const double v = std::pow(base, exponent);
            Number res(lhs.node(), rhs.node(), v);
            res.left_derivative() = exponent * std::pow(base, exponent - 1.0);
            res.right_derivative() = v * std::log(base);
            return res;
        }

        inline friend Number pow(const Number& lhs, const double& rhs) {
            const double v = std::pow(lhs.value(), rhs);
            Number res(lhs.node(), v);
            res.derivative() = rhs * std::pow(lhs.value(), rhs - 1.0);
            return res;
        }

        inline friend Number pow(const double& lhs, const Number& rhs) {
            const double v = std::pow(lhs, rhs.value());
            Number res(rhs.node(), v);
            res.derivative() = v * std::log(lhs);
            return res;
        }

        inline friend Number max(const Number& lhs, const Number& rhs) {
            const double lv = lhs.value();
            const double rv = rhs.value();
            const double v = std::max(lv, rv);
            Number res(lhs.node(), rhs.node(), v);
            res.left_derivative() = (lv > rv) ? 1.0 : 0.0;
            res.right_derivative() = (rv > lv) ? 1.0 : 0.0;
            return res;
        }

        inline friend Number max(const Number& lhs, const double& rhs) {
            const double lv = lhs.value();
            const double v = std::max(lv, rhs);
            Number res(lhs.node(), v);
            res.derivative() = (lv > rhs) ? 1.0 : 0.0;
            return res;
        }

        inline friend Number max(const double& lhs, const Number& rhs) {
            return max(rhs, lhs);
        }

        inline friend Number min(const Number& lhs, const Number& rhs) {
            const double lv = lhs.value();
            const double rv = rhs.value();
            const double v = std::min(lv, rv);
            Number res(lhs.node(), rhs.node(), v);
            res.left_derivative() = (lv < rv) ? 1.0 : 0.0;
            res.right_derivative() = (rv < lv) ? 1.0 : 0.0;
            return res;
        }

        inline friend Number min(const Number& lhs, const double& rhs) {
            const double lv = lhs.value();
            const double v = std::min(lv, rhs);
            Number res(lhs.node(), v);
            res.derivative() = (lv < rhs) ? 1.0 : 0.0;
            return res;
        }

        inline friend Number min(const double& lhs, const Number& rhs) {
            return min(rhs, lhs);
        }


        // Integer exponent specialization: pow(Number, int)
        inline friend Number pow(const Number& lhs, int rhs) {
            const double v = std::pow(lhs.value(), static_cast<double>(rhs));
            Number res(lhs.node(), v);
            res.derivative() = static_cast<double>(rhs) * std::pow(lhs.value(), rhs - 1);
            return res;
        }

        Number& operator*=(const Number& arg) {
            *this = *this * arg;
            return *this;
        }

        Number& operator+=(const Number& arg) {
            *this = *this + arg;
            return *this;
        }

        Number& operator-=(const Number& arg) {
            *this = *this - arg;
            return *this;
        }

        Number& operator/=(const Number& arg) {
            *this = *this / arg;
            return *this;
        }

        Number& operator*=(const double& arg) {
            *this = *this * arg;
            return *this;
        }

        Number& operator+=(const double& arg) {
            *this = *this + arg;
            return *this;
        }

        Number& operator-=(const double& arg) {
            *this = *this - arg;
            return *this;
        }

        Number& operator/=(const double& arg) {
            *this = *this / arg;
            return *this;
        }

        Number operator-() const {
            return 0.0 - *this;
        }

        Number operator+() const {
            return *this;
        }

        inline friend Number exp(const Number& arg) {
            const double v = std::exp(arg.value());
            Number res(arg.node(), v);
            res.derivative() = v;
            return res;
        }

        inline friend Number log(const Number& arg) {
            const double v = std::log(arg.value());
            Number res(arg.node(), v);
            res.derivative() = 1 / arg.value();
            return res;
        }

        inline friend Number sqrt(const Number& arg) {
            const double v = std::sqrt(arg.value());
            Number res(arg.node(), v);
            res.derivative() = 1 / v;
            return res;
        }

        inline friend Number fabs(const Number& arg) {
            const double v = std::fabs(arg.value());
            Number res(arg.node(), v);
            res.derivative() = arg.value() > 0 ? 1.0 : -1.0;
            return res;
        }

        inline friend bool operator==(const Number& lhs, const Number& rhs) {
            return lhs.value() == rhs.value();
        }

        inline friend bool operator==(const Number& lhs, const double& rhs) {
            return lhs.value() == rhs;
        }

        inline friend bool operator==(const double& lhs, const Number& rhs) {
            return lhs == rhs.value();
        }

        inline friend bool operator!=(const Number& lhs, const Number& rhs) {
            return lhs.value() != rhs.value();
        }

        inline friend bool operator!=(const Number& lhs, const double& rhs) {
            return lhs.value() != rhs;
        }

        inline friend bool operator!=(const double& lhs, const Number& rhs) {
            return lhs != rhs.value();
        }

        inline friend bool operator>(const Number& lhs, const Number& rhs) {
            return lhs.value() > rhs.value();
        }

        inline friend bool operator>(const Number& lhs, const double& rhs) {
            return lhs.value() > rhs;
        }

        inline friend bool operator>(const double& lhs, const Number& rhs) {
            return lhs > rhs.value();
        }

        inline friend bool operator>=(const Number& lhs, const Number& rhs) {
            return lhs.value() >= rhs.value();
        }

        inline friend bool operator>=(const Number& lhs, const double& rhs) {
            return lhs.value() >= rhs;
        }

        inline friend bool operator>=(const double& lhs, const Number& rhs) {
            return lhs >= rhs.value();
        }

        inline friend bool operator<(const Number& lhs, const Number& rhs) {
            return lhs.value() < rhs.value();
        }

        inline friend bool operator<(const Number& lhs, const double& rhs) {
            return lhs.value() < rhs;
        }

        inline friend bool operator<(const double& lhs, const Number& rhs) {
            return lhs < rhs.value();
        }

        inline friend bool operator<=(const Number& lhs, const Number& rhs) {
            return lhs.value() <= rhs.value();
        }

        inline friend bool operator<=(const Number& lhs, const double& rhs) {
            return lhs.value() <= rhs;
        }

        inline friend bool operator<=(const double& lhs, const Number& rhs) {
            return lhs <= rhs.value();
        }

        explicit operator double& () {return value_;}
        explicit operator double() const {return value_;}

    private:
        Node& node() const {return const_cast<Node&>(*node_);}

        double& derivative() {return node_->derivatives_[0];}
        double& left_derivative(){return node_->derivatives_[0];}
        double& right_derivative(){return node_->derivatives_[1];}

        double*& adjoint_ptr(){return node_->arg_adjoint_ptr_[0];}
        double*& left_adjoint_ptr(){return node_->arg_adjoint_ptr_[0];}
        double*& right_adjoint_ptr(){return node_->arg_adjoint_ptr_[1];}

        template <size_t N>
        void create_node() {
            node_ = tape->record_node<N>();
        }

        Number(Node& arg, const double val) : value_(val) {
            create_node<1>();
            node_->arg_adjoint_ptr_[0] = &arg.adjoint_;
        }

        Number(Node& lhs, Node& rhs, const double val) : value_(val) {
            create_node<2>();
            node_->arg_adjoint_ptr_[0] = &lhs.adjoint_;
            node_->arg_adjoint_ptr_[1] = &rhs.adjoint_;
        }
};
