// This code is a modified version of the code found in Antoine Savine's book "Modern Computational Finance - AAD and Parallel Simulations".
#pragma once
#include "blocklist.hpp"
#include "aadnode.hpp"

constexpr size_t BLOCKSIZE = 16384;
constexpr size_t DATASIZE = 65536;

class Tape {
    BlockList<Node, BLOCKSIZE> nodes_;
    BlockList<double, DATASIZE> derivatives_;
    BlockList<double*, DATASIZE> argptr_;

    // Padding to avoid interference for vectorised tapes
    char _pad[64];

    public:
        template<size_t N>
        Node* record_node(){
            Node* node = nodes_.emplace_back(N);
            if constexpr(N) {
                node->derivatives_ = derivatives_.emplace_back_multiple<N>();
                node->arg_adjoint_ptr_ = argptr_.emplace_back_multiple<N>();
            }
            return node;
        }

        void reset_adjoints() {
            for (Node& node : nodes_) {
                node.adjoint_ = 0;
            }
        }

        void clear() {
            derivatives_.clear();
            argptr_.clear();
            nodes_.clear();
        }

        void rewind() {
            derivatives_.rewind();
            argptr_.rewind();
            nodes_.rewind();
        }

        void mark() {
            derivatives_.set_mark();
            argptr_.set_mark();
            nodes_.set_mark();
        }

        void rewind_to_mark() {
            derivatives_.rewind_to_mark();
            argptr_.rewind_to_mark();
            nodes_.rewind_to_mark();
        }

        using Iterator = BlockList<Node, BLOCKSIZE>::Iterator;
        
        auto begin() {
            return nodes_.begin();
        }

        auto end() {
            return nodes_.end();
        }

        auto mark_iter() {
            return nodes_.mark();
        }

        auto find(Node* node) {
            return nodes_.find(node);
        }
};