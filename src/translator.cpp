#include "translator.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>

namespace translation {

using namespace std::chrono_literals;

BatchingTranslator::BatchingTranslator(Config cfg) : cfg_(std::move(cfg))
{
    tokenizer_ = std::make_unique<Tokenizer>(cfg_.spm_dir);

    // CTranslate2 v4: Translator handles threading internally.
    // For CPU, we can control parallelism via environment variables:
    //   CT2_INTER_THREADS = num_replicas (model copies)
    //   CT2_INTRA_THREADS = threads per replica
    // Or load the model first, then create Translator from it.
    translator_ = std::make_unique<ctranslate2::Translator>(
        cfg_.model_path,
        ctranslate2::Device::CPU,
        // 0,  // device_index (ignored for CPU)
        ctranslate2::ComputeType::DEFAULT
    );

    // Start the dispatcher after all members are initialised.
    dispatcher_ = std::thread(&BatchingTranslator::dispatcher_loop, this);
}

BatchingTranslator::~BatchingTranslator() {
    stop_.store(true, std::memory_order_release);
    queue_cv_.notify_one();
    if (dispatcher_.joinable())
        dispatcher_.join();
}

std::future<std::vector<std::string>>
BatchingTranslator::translate_async(std::vector<std::string> texts) {
    WorkItem item;
    item.texts = std::move(texts);
    auto fut = item.promise.get_future();

    {
        std::lock_guard lock{queue_mtx_};
        queue_.push_back(std::move(item));
    }
    queue_cv_.notify_one();
    return fut;
}

// ── Dispatcher ──────────────────────────────────────────────────────────────
// Runs on its own thread.  Collects work items until either:
//   (a) max_batch_size items are queued, OR
//   (b) batch_timeout_us microseconds have elapsed since the first item arrived.
// Then hands the full batch to CT2 synchronously.
void BatchingTranslator::dispatcher_loop() {
    const auto timeout = std::chrono::microseconds(cfg_.batch_timeout_us);

    while (true) {
        std::vector<WorkItem> batch;

        {
            std::unique_lock lock{queue_mtx_};

            // Block until at least one item arrives or stop is requested.
            queue_cv_.wait(lock, [this] {
                return !queue_.empty() || stop_.load(std::memory_order_acquire);
            });

            if (stop_.load(std::memory_order_acquire) && queue_.empty())
                break;

            // Drain up to max_batch_size, waiting a short extra window to
            // coalesce concurrent requests into one big CT2 call.
            const auto deadline = std::chrono::steady_clock::now() + timeout;

            while (batch.size() < cfg_.max_batch_size) {
                if (queue_.empty()) {
                    if (std::chrono::steady_clock::now() >= deadline)
                        break;
                    // Briefly release the lock so IO threads can push more.
                    lock.unlock();
                    std::this_thread::sleep_for(200us);
                    lock.lock();
                    continue;
                }
                batch.push_back(std::move(queue_.front()));
                queue_.pop_front();
            }
        }

        if (!batch.empty())
            run_batch(std::move(batch));
    }
}

// ── Batch runner ─────────────────────────────────────────────────────────────
// Flattens all texts from all WorkItems into one CT2 call for maximum
// hardware utilisation, then splits results back per-item.
void BatchingTranslator::run_batch(std::vector<WorkItem> batch) {
    // 1. Build the flat index — record where each item's texts start/end.
    std::vector<size_t> offsets;
    offsets.reserve(batch.size() + 1);
    offsets.push_back(0);

    std::vector<std::string> flat_texts;
    for (const auto& item : batch) {
        for (const auto& t : item.texts)
            flat_texts.push_back(t);
        offsets.push_back(flat_texts.size());
    }

    try {
        // 2. Tokenize the whole flat batch in one pass.
        auto token_batches = tokenizer_->encode_batch(flat_texts);

        // 3. Translate — CT2 v4 uses translator_->translate_batch()
        ctranslate2::TranslationOptions opts;
        opts.beam_size           = static_cast<int>(cfg_.beam_size);
        opts.max_decoding_length = static_cast<int>(cfg_.max_decoding_length);
        // No length penalty needed for greedy; keep it tight.
        opts.no_repeat_ngram_size = 0;

        const auto ct2_results = translator_->translate_batch(token_batches, opts);

        // 4. Decode — extract hypothesis[0] (best beam) for every result.
        std::vector<std::vector<std::string>> decoded_tokens;
        decoded_tokens.reserve(ct2_results.size());
        for (const auto& r : ct2_results) {
            if (r.hypotheses.empty())
                decoded_tokens.push_back({});
            else
                decoded_tokens.push_back(r.hypotheses[0]);
        }

        auto decoded_strings = tokenizer_->decode_batch(decoded_tokens);

        // 5. Fan results back to each waiting promise.
        for (size_t i = 0; i < batch.size(); ++i) {
            const size_t start = offsets[i];
            const size_t end   = offsets[i + 1];
            std::vector<std::string> item_results(
                decoded_strings.begin() + static_cast<ptrdiff_t>(start),
                decoded_strings.begin() + static_cast<ptrdiff_t>(end)
            );
            batch[i].promise.set_value(std::move(item_results));
        }

    } catch (...) {
        // Propagate the exception to every waiting caller.
        auto eptr = std::current_exception();
        for (auto& item : batch)
            item.promise.set_exception(eptr);
    }
}

} // namespace translation