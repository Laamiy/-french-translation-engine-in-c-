#pragma once

#include "tokenizer.hpp"

#include <ctranslate2/translator.h>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace translation {

class BatchingTranslator 
    {
        public:
            struct Config 
            {
                std::string model_path;
                std::string spm_dir;
                size_t      num_workers        = 4;    // Ignored in v4 — use env vars CT2_INTER_THREADS
                size_t      intra_op_threads   = 2;    // Ignored in v4 — use env vars CT2_INTRA_THREADS
                size_t      max_batch_size     = 64;
                uint32_t    batch_timeout_us   = 5000;
                size_t      beam_size          = 1;
                size_t      max_decoding_length = 128;
            };

            explicit BatchingTranslator(Config cfg);
            ~BatchingTranslator();

            BatchingTranslator(const BatchingTranslator&)            = delete;
            BatchingTranslator& operator=(const BatchingTranslator&) = delete;

            [[nodiscard]] std::future<std::vector<std::string>>
            translate_async(std::vector<std::string> texts);

        private:
            struct WorkItem 
            {
                std::vector<std::string>                texts;
                std::promise<std::vector<std::string>>      promise;
            };

            void dispatcher_loop();
            void run_batch(std::vector<WorkItem> batch);

            Config                    cfg_;
            std::unique_ptr<Tokenizer>                      tokenizer_;
            std::unique_ptr<ctranslate2::Translator>        translator_;  // for v3:  TranslatorPool

            std::mutex               queue_mtx_;
            std::condition_variable  queue_cv_;
            std::deque<WorkItem>     queue_;
            std::atomic<bool>        stop_{false};
            std::thread              dispatcher_;
        };

} // namespace translation