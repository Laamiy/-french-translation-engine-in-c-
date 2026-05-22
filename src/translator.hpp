#pragma once

#include "tokenizer.hpp"

#include <onnxruntime_cxx_api.h>

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

class BatchingTranslator {
public:
    struct Config {
        std::string model_path;   // path to onnx model dir (contains encoder_model.onnx, decoder_model.onnx, etc.)
        std::string spm_dir;      // path to dir containing source.spm / target.spm
        size_t      num_workers        = 4;
        size_t      intra_op_threads   = 2;
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
    struct WorkItem {
        std::vector<std::string>                    texts;
        std::promise<std::vector<std::string>>      promise;
    };

    void dispatcher_loop();
    void run_batch(std::vector<WorkItem> batch);

    // ONNX inference for a single text
    std::string translate_one(const std::string& text);

    Config                    cfg_;
    std::unique_ptr<Tokenizer>                      tokenizer_;
    
    // ONNX Runtime
    Ort::Env                ort_env_;
    Ort::SessionOptions     ort_session_options_;
    std::unique_ptr<Ort::Session> encoder_session_;
    std::unique_ptr<Ort::Session> decoder_session_;

    std::mutex               queue_mtx_;
    std::condition_variable  queue_cv_;
    std::deque<WorkItem>     queue_;
    std::atomic<bool>        stop_{false};
    std::thread              dispatcher_;
};

} // namespace translation