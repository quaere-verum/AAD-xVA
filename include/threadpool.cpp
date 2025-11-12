// This code is a modified version of the code found in Antoine Savine's book "Modern Computational Finance - AAD and Parallel Simulations".
#include "threadpool.hpp"

ThreadPool ThreadPool::pool_;

thread_local std::size_t ThreadPool::thread_num_ = 0;