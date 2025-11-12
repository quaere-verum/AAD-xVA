// This code is a modified version of the code found in Antoine Savine's book "Modern Computational Finance - AAD and Parallel Simulations".
#include <array>
#include <list>
#include <iterator>

template <class T, size_t block_size>
class BlockList {
    std::list<std::array<T, block_size>> data;
    using list_iter = decltype(data.begin());
    using block_iter = decltype(data.back().begin());

    list_iter current_block;
    list_iter last_block;
    list_iter marked_block;
    block_iter next_space;
    block_iter last_space;
    block_iter marked_space;

    private:
        void new_block(){
            data.emplace_back();
            current_block = last_block = std::prev(data.end());
            next_space = current_block->begin();
            last_space = current_block->end();
        }

        void next_block(){
            if (current_block == last_block){
                new_block();
            }
            else {
                ++current_block;
                next_space = current_block->begin();
                last_space = current_block->end();
            }
        }
    
    public:
        BlockList() {
            new_block();
        }

        void clear() {
            data.clear();
            new_block();
        }

        void rewind() {
            current_block = data.begin();
            next_space = current_block->begin();
            last_space = current_block->end();
        }

        void set_mark() {
            marked_block = current_block;
            marked_space = next_space;
        }

        void rewind_to_mark() {
            current_block = marked_block;
            next_space = marked_space;
            last_space = current_block->end();
        }

        T* emplace_back() {
            if (next_space == last_space) next_block();
            auto old_next = next_space;
            ++next_space;
            return &*old_next;
        }

        template<size_t n>
        T* emplace_back_multiple() {
            if (std::distance(next_space, last_space) < n) next_block();
            auto old_next = next_space;
            next_space += n;
            return &*old_next;
        }

        T* emplace_back_multiple(const size_t n) {
            if (std::distance(next_space, last_space) < n) next_block();
            auto old_next = next_space;
            next_space += n;
            return &*old_next;
        }

        void memset(unsigned char value = 0) {
            for (auto& arr : data) {
                std::memset(&arr[0], value, block_size * sizeof(T));
            }
        }

        template<typename ...Args>
        T* emplace_back(Args&& ...args) {
            if (next_space == last_space) next_block();
            T* emplaced = new (&*next_space) T(std::forward<Args>(args)...);
            ++next_space;
            return emplaced;
        }

        class Iterator {
            list_iter current_block;
            block_iter current_space;
            block_iter first_space;
            block_iter last_space;

            public:
                using difference_type = ptrdiff_t;
                using reference = T&;
                using pointer = T*;
                using value_type = T;
                using iterator_category = std::bidirectional_iterator_tag;

                Iterator() {};

                Iterator(list_iter cb, block_iter cs, block_iter fs, block_iter ls) :
                    current_block(cb), current_space(cs), first_space(fs), last_space(ls) {};
                Iterator& operator++() {
                    ++current_space;
                    if (current_space == last_space) {
                        ++current_block;
                        first_space = current_block->begin();
                        last_space = current_block->end();
                        current_space = first_space;
                    }
                    return *this;
                }

                Iterator& operator--() {
                    if (current_space == first_space) {
                        --current_block;
                        first_space = current_block->begin();
                        last_space = current_block->end();
                        current_space = last_space;
                    }
                    --current_space;
                    return *this;
                }

                T& operator*() {
                    return *current_space;
                }

                const T& operator*() const {
                    return *current_space;
                }

                T* operator->() {
                    return &*current_space;
                }

                const T* operator->() const {
                    return &*current_space;
                }

                bool operator==(const Iterator& rhs) const {
                    return (current_block == rhs.current_block && current_space == rhs.current_space);
                }

                bool operator!=(const Iterator& rhs) const {
                    return (current_block != rhs.current_block || current_space != rhs.current_space);
                }
        };

        Iterator begin() {
            return Iterator(data.begin(), data.begin()->begin(), data.begin()->begin(), data.begin()->end());
        }

        Iterator end() {
            auto last_block = std::prev(data.end());
            return Iterator(current_block, next_space, current_block->begin(), current_block->end());
        }

        Iterator mark() {
            return Iterator(marked_block, marked_space, marked_block->begin(), marked_block->end());
        }

        Iterator find(const T* const element) {
            Iterator it = end();
            Iterator b = begin();

            while (it != b) {
                --it;
                if (&*it == element) return it;
            }

            if (&*it == element) return it;
            
            return end();
        }

};

