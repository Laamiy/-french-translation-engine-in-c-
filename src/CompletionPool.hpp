#pragma once 
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <future>
#include <queue>

namespace translation
{

    class CompletionPool 
    {
        public:
            explicit CompletionPool(size_t n) 
            {
                workers_.reserve(n);
                for (size_t i = 0; i < n; ++i)
                    workers_.emplace_back([this] { run(); });
            }

            ~CompletionPool() 
            {
                {
                    std::lock_guard lock{mtx_};
                    stop_ = true;
                }
                cv_.notify_all();
                for (auto& t : workers_) if (t.joinable()) t.join();
            }

            void post(std::function<void()> task) {
                {
                    std::lock_guard lock{mtx_};
                    queue_.push(std::move(task));
                }
                cv_.notify_one();
            }

        private:
            void run() 
            {
                while (true) 
                {
                    std::function<void()> task;
                    {
                        std::unique_lock lock{mtx_};
                        cv_.wait(lock, [this] { return !queue_.empty() || stop_; });
                        if (stop_ && queue_.empty()) return;
                        task = std::move(queue_.front());
                        queue_.pop();
                    }
                    task();
                }
            }

            std::vector<std::thread>       workers_;
            std::queue<std::function<void()>> queue_;
            std::mutex                     mtx_;
            std::condition_variable        cv_;
            bool                           stop_{false};
        };
};
