// This code is a modified version of the code found in Antoine Savine's book "Modern Computational Finance - AAD and Parallel Simulations".
#include <thread>
#include <future>
#include <vector>
#include <chrono>
#include "concurrentq.hpp"

using Task = std::packaged_task<bool(void)>;
using TaskHandle = std::future<bool>;

class ThreadPool {
    static ThreadPool pool_;
    ConcurrentQueue<Task> q;
    std::vector<std::thread> threads;
    bool active_;
    bool interrupt_;
    static thread_local std::size_t thread_num_;

    void thread_func(const std::size_t n) {
        thread_num_ = n;
        Task t;
        while (!interrupt_) {
            q.pop(t);
            if (!interrupt_) t();
        }
    };

    ThreadPool() : active_(false), interrupt_(false) {};

    public:
        static ThreadPool* get_pool() {return &pool_;};

        std::size_t n_threads() const {return threads.size();};

        static std::size_t thread_num() {return thread_num_;};

        void start(const std::size_t n_threads_ = std::thread::hardware_concurrency() - 1) {
            if (!active_) {
                threads.reserve(n_threads_);
                for (std::size_t i = 0; i < n_threads_; i++) {
                    threads.push_back(std::thread(&ThreadPool::thread_func, this, i + 1));
                }
                active_ = true;
            }
        };

        void stop(){
            if (active_){
                interrupt_ = true;
                q.interrupt();
                std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
                threads.clear();
                q.clear();
                q.reset_interrupt();
                active_ = false;
                interrupt_ = false;
            }
        };

        ~ThreadPool(){
            stop();
        }

        ThreadPool(const ThreadPool& rhs) = delete;
        ThreadPool& operator=(const ThreadPool& rhs) = delete;
        ThreadPool(ThreadPool&& rhs) = delete;
        ThreadPool& operator=(ThreadPool&& rhs) = delete;

        template<typename Callable>
        TaskHandle spawn_task(Callable c) {
            Task t(std::move(c));
            TaskHandle f = t.get_future();
            q.push(std::move(t));
            return f;
        };

        bool active_wait(const TaskHandle& f) {
            Task t;
            bool b = false;
            while (f.wait_for(std::chrono::seconds(0)) != std::future_status::ready){
                if (q.try_pop(t)) {
                    t();
                    b = true;
                }
                else {
                    f.wait();
                }
            };
            return b;
        };
};