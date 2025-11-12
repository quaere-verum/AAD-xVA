// This code is a modified version of the code found in Antoine Savine's book "Modern Computational Finance - AAD and Parallel Simulations".
#include <queue>
#include <mutex>
#include <condition_variable>

template <class T>
class ConcurrentQueue {
    std::queue<T> q;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool interrupt_;

    public:
        ConcurrentQueue() : interrupt_(false) {}
        ~ConcurrentQueue() {interrupt();};

        bool empty() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return q.empty();
        };

        bool try_pop(T& t) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (q.empty()) return false;
            t = std::move(q.front());
            q.pop();
            return true;
        };

        void push(T t) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                q.push(std::move(t));
            }
            cv_.notify_one();
        };

        bool pop(T& t) {
            std::unique_lock<std::mutex> lock(mutex_);
            while (!interrupt_ && q.empty()) cv_.wait(lock);
            if (interrupt_) return false;
            t = std::move(q.front());
            q.pop();
            return true;
        };

        void interrupt() {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                interrupt_ = true;
            }
            cv_.notify_all();
        };

        void reset_interrupt() {
            interrupt_ = false;
        };

        void clear() {
            std::queue<T> empty;
            std::swap(q, empty);
        };

};