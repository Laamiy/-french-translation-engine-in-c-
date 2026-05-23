    #pragma once

    #include "tokenizer.hpp"

    #include <onnxruntime_cxx_api.h>

    #include <atomic>
    #include <condition_variable>
    #include <deque>
    #include <future>
    #include <mutex>
    #include <string>
    #include <thread>
    #include <vector>

    namespace translation {

    // Number of transformer layers — matches your model's present.N.* outputs.
    static constexpr int kNumLayers = 6;
    static constexpr int kNumHeads  = 8;
    static constexpr int kHeadDim   = 64;

    class BatchingTranslator {
    public:
        struct Config {
            std::string model_path;             // dir with encoder/decoder onnx files
            std::string spm_dir;                // dir with source.spm / target.spm
            size_t      num_workers         = 4;
            size_t      intra_op_threads    = 2;
            size_t      max_batch_size      = 32;
            uint32_t    batch_timeout_us    = 5000;
            size_t      max_decoding_length = 128;
            // Token IDs from generation_config.json
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

        // Per-sentence KV cache state, carried across decoder steps.
        struct KVCache {
            // encoder KV: fixed after first decoder step — shape [1, heads, enc_len, head_dim]
            // decoder KV: grows each step — shape [1, heads, step, head_dim]
            // Stored flat: [layer][key|value]
            std::vector<std::vector<float>> enc_key;   // [kNumLayers]
            std::vector<std::vector<float>> enc_val;
            std::vector<std::vector<float>> dec_key;
            std::vector<std::vector<float>> dec_val;

            std::vector<int64_t> enc_kv_shape; // [1, heads, enc_len, head_dim]
            std::vector<int64_t> dec_kv_shape; // updated each step

            bool encoder_kv_initialized = false;
        };

        void        dispatcher_loop();
        void        run_batch(std::vector<WorkItem> batch);
        std::string translate_one(const std::string& text);

        // Runs encoder, returns hidden states + attention mask (both needed by decoder).
        std::vector<float> run_encoder(
            const std::vector<int64_t>& input_ids,
            std::vector<int64_t>&       out_attention_mask,
            std::vector<int64_t>&       out_enc_shape
        );

        // First decoder step — no past KV supplied; produces logits + initialises cache.
        int64_t decoder_first_step(
            int64_t                    start_token,
            const std::vector<float>&  encoder_hidden,
            const std::vector<int64_t>&enc_shape,
            const std::vector<int64_t>&enc_mask,
            KVCache&                   cache
        );

        // Subsequent decoder steps — uses decoder_with_past session.
        int64_t decoder_step_with_past(
            int64_t                    last_token,
            const std::vector<float>&  encoder_hidden,
            const std::vector<int64_t>&enc_shape,
            const std::vector<int64_t>&enc_mask,
            KVCache&                   cache
        );

        Config cfg_;

        std::unique_ptr<Tokenizer> tokenizer_;

        // One shared OrtEnv — cheap, thread-safe.
        Ort::Env            ort_env_;
        Ort::SessionOptions session_opts_;

        std::unique_ptr<Ort::Session> enc_session_;
        std::unique_ptr<Ort::Session> dec_session_;       // decoder_model.onnx (no past)
        std::unique_ptr<Ort::Session> dec_wp_session_;    // decoder_with_past_model.onnx

        // Batching queue
        std::mutex              queue_mtx_;
        std::condition_variable queue_cv_;
        std::deque<WorkItem>    queue_;
        std::atomic<bool>       stop_{false};

        // N dispatcher threads for parallel sentence inference.
        std::vector<std::thread> dispatchers_;
    };

    } // namespace translation