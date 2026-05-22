#include "translator.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <filesystem>

namespace translation {

using namespace std::chrono_literals;
namespace fs = std::filesystem;

// Helper: Get input/output names from ONNX session
static std::vector<const char*> get_input_names(const Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t count = session.GetInputCount();
    std::vector<const char*> names;
    for (size_t i = 0; i < count; i++) {
        names.push_back(session.GetInputNameAllocated(i, allocator).get());
    }
    return names;
}

static std::vector<const char*> get_output_names(const Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t count = session.GetOutputCount();
    std::vector<const char*> names;
    for (size_t i = 0; i < count; i++) {
        names.push_back(session.GetOutputNameAllocated(i, allocator).get());
    }
    return names;
}

BatchingTranslator::BatchingTranslator(Config cfg) : cfg_(std::move(cfg)), ort_env_(ORT_LOGGING_LEVEL_WARNING, "translation_server")
{
    tokenizer_ = std::make_unique<Tokenizer>(cfg_.spm_dir);

    // Configure ONNX Runtime
    ort_session_options_.SetIntraOpNumThreads(static_cast<int>(cfg_.intra_op_threads));
    ort_session_options_.SetInterOpNumThreads(static_cast<int>(cfg_.num_workers));
    ort_session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    const fs::path model_dir{cfg_.model_path};
    const std::string encoder_path = (model_dir / "encoder_model.onnx").string();
    const std::string decoder_path = (model_dir / "decoder_model.onnx").string();

    encoder_session_ = std::make_unique<Ort::Session>(ort_env_, encoder_path.c_str(), ort_session_options_);
    decoder_session_ = std::make_unique<Ort::Session>(ort_env_, decoder_path.c_str(), ort_session_options_);

    // Start the dispatcher
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

void BatchingTranslator::dispatcher_loop() {
    const auto timeout = std::chrono::microseconds(cfg_.batch_timeout_us);

    while (true) {
        std::vector<WorkItem> batch;

        {
            std::unique_lock lock{queue_mtx_};

            queue_cv_.wait(lock, [this] {
                return !queue_.empty() || stop_.load(std::memory_order_acquire);
            });

            if (stop_.load(std::memory_order_acquire) && queue_.empty())
                break;

            const auto deadline = std::chrono::steady_clock::now() + timeout;

            while (batch.size() < cfg_.max_batch_size) {
                if (queue_.empty()) {
                    if (std::chrono::steady_clock::now() >= deadline)
                        break;
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

std::string BatchingTranslator::translate_one(const std::string& text) {
    // 1. Tokenize
    auto tokens = tokenizer_->encode_batch({text})[0];
    
    // 2. Convert tokens to IDs using source SPM
    // For Marian, we need to map tokens to IDs. Since SPM gives us string tokens,
    // we need to look them up. But our Tokenizer only does SPM encode/decode.
    // For ONNX, we need integer input IDs.
    
    // Simplified: Use SPM to get IDs
    // This is a placeholder - you'll need to implement proper ID mapping
    // or use the vocab.json from the ONNX model
    
    // For now, return dummy to compile
    return "[ONNX translation placeholder for: " + text + "]";
}

void BatchingTranslator::run_batch(std::vector<WorkItem> batch) {
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
        // Translate each text individually (batching ONNX is complex)
        std::vector<std::string> translations;
        translations.reserve(flat_texts.size());
        
        for (const auto& text : flat_texts) {
            translations.push_back(translate_one(text));
        }

        // Fan results back
        for (size_t i = 0; i < batch.size(); ++i) {
            const size_t start = offsets[i];
            const size_t end   = offsets[i + 1];
            std::vector<std::string> item_results(
                translations.begin() + static_cast<ptrdiff_t>(start),
                translations.begin() + static_cast<ptrdiff_t>(end)
            );
            batch[i].promise.set_value(std::move(item_results));
        }

    } catch (...) {
        auto eptr = std::current_exception();
        for (auto& item : batch)
            item.promise.set_exception(eptr);
    }
}

} // namespace translation