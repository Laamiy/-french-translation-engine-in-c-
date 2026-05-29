#pragma once

#include "tokenizer.hpp"
#include <onnxruntime_cxx_api.h>

#include <atomic>
#include <array>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace translation {

static constexpr int kNumLayers = 6;
static constexpr int kNumHeads  = 8;
static constexpr int kHeadDim   = 64;

class BatchingTranslator {
public:
    struct Config {
        std::string model_path;
        std::string spm_dir;
        size_t      num_workers         = 4;
        size_t      intra_op_threads    = 2;
        size_t      max_batch_size      = 32;
        uint32_t    batch_timeout_us    = 5000;
        size_t      max_decoding_length = 128;
        int64_t     decoder_start_token = 59513;
        int64_t     eos_token_id        = 0;
        int64_t     pad_token_id        = 59513;
    };

    explicit BatchingTranslator(Config cfg);
    ~BatchingTranslator();

    BatchingTranslator(const BatchingTranslator&)            = delete;
    BatchingTranslator& operator=(const BatchingTranslator&) = delete;

    [[nodiscard]] std::future<std::vector<std::string>>
    translate_async(std::vector<std::string> texts);

    size_t num_workers() const { return cfg_.num_workers; }

private:
    struct WorkItem {
        std::vector<std::string>               texts;
        std::promise<std::vector<std::string>> promise;
    };

    // Per-thread KV cache — one instance lives on each dispatcher thread's
    // stack for the lifetime of the thread. Buffers are reused across
    // requests via assign(), eliminating per-request heap allocation.
    struct KVCache {
        std::vector<std::vector<float>> enc_key;  // [kNumLayers]
        std::vector<std::vector<float>> enc_val;
        std::vector<std::vector<float>> dec_key;
        std::vector<std::vector<float>> dec_val;
        std::vector<int64_t>            enc_kv_shape;
        std::vector<int64_t>            dec_kv_shape;
        bool encoder_kv_initialized = false;
    };

    // Core inference path — cache passed by ref from dispatcher thread.
    std::string translate_one(const std::string& text, KVCache& cache);

    std::vector<float> run_encoder(
        const std::vector<int64_t>& input_ids,
        std::vector<int64_t>&       out_mask,
        std::vector<int64_t>&       out_enc_shape
    );

    int64_t decoder_first_step(
        int64_t                     start_token,
        const std::vector<float>&   enc_hidden,
        const std::vector<int64_t>& enc_shape,
        const std::vector<int64_t>& enc_mask,
        KVCache&                    cache
    );

    // Simplified signature — enc_hidden/enc_shape not needed after first step.
    int64_t decoder_step_with_past(
        int64_t                     last_token,
        const std::vector<int64_t>& enc_mask,
        KVCache&                    cache
    );

    void dispatcher_loop();
    void run_batch(std::vector<WorkItem> batch);

    Config                     cfg_;
    std::unique_ptr<Tokenizer> tokenizer_;

    Ort::Env            ort_env_;
    Ort::SessionOptions session_opts_;
    Ort::MemoryInfo     mem_info_;

    std::unique_ptr<Ort::Session> enc_session_;
    std::unique_ptr<Ort::Session> dec_session_;
    std::unique_ptr<Ort::Session> dec_wp_session_;

    // Pre-built I/O name arrays — constructed once, reused every decode step.
    std::vector<std::string>  dec_first_out_strs_;
    std::vector<const char*>  dec_first_out_names_;
    std::vector<std::string>  dec_wp_in_strs_;
    std::vector<const char*>  dec_wp_in_names_;
    std::vector<std::string>  dec_wp_out_strs_;
    std::vector<const char*>  dec_wp_out_names_;

    std::mutex               queue_mtx_;
    std::condition_variable  queue_cv_;
    std::deque<WorkItem>     queue_;
    std::atomic<bool>        stop_{false};
    std::vector<std::thread> dispatchers_;
};

} // namespace translation