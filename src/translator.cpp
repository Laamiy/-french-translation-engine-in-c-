#include "translator.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace translation {

using namespace std::chrono_literals;
namespace fs = std::filesystem;

// ── Construction ─────────────────────────────────────────────────────────────

BatchingTranslator::BatchingTranslator(Config cfg)
    : cfg_(std::move(cfg))
    , ort_env_(ORT_LOGGING_LEVEL_WARNING, "translation_server")
{
    tokenizer_ = std::make_unique<Tokenizer>(cfg_.spm_dir);

    // Session options shared across all three sessions.
    // intra_op controls GEMM thread count inside a single Run() call.
    // inter_op = 1: we manage parallelism ourselves via dispatcher threads.
    session_opts_.SetIntraOpNumThreads(static_cast<int>(cfg_.intra_op_threads));
    session_opts_.SetInterOpNumThreads(1);
    session_opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_opts_.EnableMemPattern();
    session_opts_.EnableCpuMemArena();

    const fs::path dir{cfg_.model_path};
    auto load = [&](const std::string& name) 
    {
        const std::string path = (dir / name).string();

        if (!fs::exists(path))
            throw std::runtime_error("ONNX model not found: " + path);
        std::cout << "  Loading " << path << "\n";
        return std::make_unique<Ort::Session>(ort_env_, path.c_str(), session_opts_);
    };

    std::cout << "Loading ONNX sessions...\n";
    enc_session_    = load("encoder_model.onnx");
    dec_session_    = load("decoder_model.onnx");
    dec_wp_session_ = load("decoder_with_past_model.onnx");
    std::cout << "All sessions loaded.\n";

    // Spawn num_workers dispatcher threads — each pulls work independently,
    // so N sentences can be running through encoder+decoder concurrently.
    // Each thread calls ORT with intra_op_threads, so total CPU =
    // num_workers * intra_op_threads. Set product == physical core count.
    dispatchers_.reserve(cfg_.num_workers);
    for (size_t i = 0; i < cfg_.num_workers; ++i)
        dispatchers_.emplace_back(&BatchingTranslator::dispatcher_loop, this);
}

BatchingTranslator::~BatchingTranslator() 
{
    stop_.store(true, std::memory_order_release);
    queue_cv_.notify_all(); // wake all dispatcher threads
    for (auto& t : dispatchers_)
        if (t.joinable()) 
        t.join();
}

// ── Public API ────────────────────────────────────────────────────────────────

std::future<std::vector<std::string>>
BatchingTranslator::translate_async(std::vector<std::string> texts) {
    WorkItem item;
    item.texts = std::move(texts);
    auto fut   = item.promise.get_future();
    {
        std::lock_guard lock{queue_mtx_};
        queue_.push_back(std::move(item));
    }
    queue_cv_.notify_one();
    return fut;
}

// ── Dispatcher
// Each dispatcher thread pulls one WorkItem at a time and processes it fully.
// This keeps things simple and gives true parallelism without any batching
// complexity — ORT's intra-op threads handle GEMM parallelism within each job.

void BatchingTranslator::dispatcher_loop() 
{
    while (true) 
    {
        WorkItem item;
        {
            std::unique_lock lock{queue_mtx_};
            queue_cv_.wait(lock, [this] 
                {
                    return !queue_.empty() || stop_.load(std::memory_order_acquire);
                });
            if (stop_.load(std::memory_order_acquire) && queue_.empty())
                return;
            item = std::move(queue_.front());
            queue_.pop_front();
        }

        // Process the item outside the lock.
        try {
            std::vector<std::string> results;
            results.reserve(item.texts.size());
            for (const auto& text : item.texts)
                results.push_back(translate_one(text));
            
            item.promise.set_value(std::move(results));
            } 
        catch (...) 
        {
            item.promise.set_exception(std::current_exception());
        }
    }
}

// ── Encoder 

std::vector<float> BatchingTranslator::run_encoder(
    const std::vector<int64_t>& input_ids,
    std::vector<int64_t>&       out_mask,
    std::vector<int64_t>&       out_enc_shape)
{
    const int64_t seq_len = static_cast<int64_t>(input_ids.size());
    const std::vector<int64_t> shape{1, seq_len};

    out_mask.assign(seq_len, 1);

    Ort::MemoryInfo mem{"Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault};

    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem,
        const_cast<int64_t*>(input_ids.data()), input_ids.size(),
        shape.data(), shape.size()
    ));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem,
        out_mask.data(), out_mask.size(),
        shape.data(), shape.size()
    ));

    const char* in_names[]  = {"input_ids", "attention_mask"};
    const char* out_names[] = {"last_hidden_state"};

    auto outputs = enc_session_->Run(
        Ort::RunOptions{nullptr},
        in_names,  inputs.data(),  2,
        out_names, 1
    );

    out_enc_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const float* ptr = outputs[0].GetTensorData<float>();
    const size_t n   = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    return std::vector<float>(ptr, ptr + n);
}

// ── Decoder helpers ───────────────────────────────────────────────────────────

// Build the fixed names for present.N.{decoder|encoder}.{key|value}
// that the decoder outputs and the decoder_with_past inputs expect.
static std::string present_name(int layer, const char* side, const char* kv) {
    return "present." + std::to_string(layer) + "." + side + "." + kv;
}
static std::string past_name(int layer, const char* side, const char* kv) {
    return "past_key_values." + std::to_string(layer) + "." + side + "." + kv;
}

// Argmax over a float array of length n — returns index of maximum.
static int64_t argmax(const float* data, int64_t n) {
    return static_cast<int64_t>(
        std::max_element(data, data + n) - data
    );
}

// ── First decoder step (no past KV) ──────────────────────────────────────────

int64_t BatchingTranslator::decoder_first_step(
    int64_t                    start_token,
    const std::vector<float>&  enc_hidden,
    const std::vector<int64_t>&enc_shape,
    const std::vector<int64_t>&enc_mask,
    KVCache&                   cache)
{
    Ort::MemoryInfo mem{"Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault};

    const int64_t enc_len = enc_shape[1];
    std::vector<int64_t> dec_ids{start_token};
    const std::vector<int64_t> dec_shape{1, 1};
    const std::vector<int64_t> enc_mask_shape{1, enc_len};

    std::vector<const char*> in_names;
    std::vector<Ort::Value>  inputs;

    // encoder_attention_mask
    in_names.push_back("encoder_attention_mask");
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem,
        const_cast<int64_t*>(enc_mask.data()), enc_mask.size(),
        enc_mask_shape.data(), enc_mask_shape.size()
    ));

    // input_ids
    in_names.push_back("input_ids");
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem, dec_ids.data(), 1,
        dec_shape.data(), dec_shape.size()
    ));

    // encoder_hidden_states
    in_names.push_back("encoder_hidden_states");
    inputs.push_back(Ort::Value::CreateTensor<float>(
        mem,
        const_cast<float*>(enc_hidden.data()), enc_hidden.size(),
        enc_shape.data(), enc_shape.size()
    ));

    // Collect output names: logits + all present KV tensors.
    std::vector<std::string> out_name_strs;
    out_name_strs.push_back("logits");
    for (int l = 0; l < kNumLayers; ++l) {
        out_name_strs.push_back(present_name(l, "decoder", "key"));
        out_name_strs.push_back(present_name(l, "decoder", "value"));
        out_name_strs.push_back(present_name(l, "encoder", "key"));
        out_name_strs.push_back(present_name(l, "encoder", "value"));
    }
    std::vector<const char*> out_names;
    out_names.reserve(out_name_strs.size());
    for (const auto& s : out_name_strs) out_names.push_back(s.c_str());

    auto outputs = dec_session_->Run(
        Ort::RunOptions{nullptr},
        in_names.data(),  inputs.data(),  in_names.size(),
        out_names.data(), out_names.size()
    );

    // Argmax on logits[0, 0, :] — shape [1, 1, vocab_size]
    const float* logits    = outputs[0].GetTensorData<float>();
    const int64_t vocab_sz = outputs[0].GetTensorTypeAndShapeInfo().GetShape()[2];
    const int64_t next_tok = argmax(logits, vocab_sz);

    // Store KV cache — present.N outputs start at index 1.
    cache.enc_key.resize(kNumLayers);
    cache.enc_val.resize(kNumLayers);
    cache.dec_key.resize(kNumLayers);
    cache.dec_val.resize(kNumLayers);

    for (int l = 0; l < kNumLayers; ++l) {
        // Order: dec_key, dec_val, enc_key, enc_val per layer
        const int base = 1 + l * 4;
        auto copy_tensor = [&](int idx) {
            const float* p = outputs[idx].GetTensorData<float>();
            const size_t n = outputs[idx].GetTensorTypeAndShapeInfo().GetElementCount();
            return std::vector<float>(p, p + n);
        };
        cache.dec_key[l] = copy_tensor(base + 0);
        cache.dec_val[l] = copy_tensor(base + 1);
        cache.enc_key[l] = copy_tensor(base + 2);
        cache.enc_val[l] = copy_tensor(base + 3);
    }

    // Save shapes for future steps.
    // dec KV shape after step 1: [1, heads, 1, head_dim]
    // cache.dec_kv_shape = {1, kNumHeads, 1, kHeadDim};
    // // enc KV shape: [1, heads, enc_len, head_dim]
    // cache.enc_kv_shape = {1, kNumHeads, enc_len, kHeadDim};
    cache.dec_kv_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape(); // present.0.decoder.key
    cache.enc_kv_shape = outputs[3].GetTensorTypeAndShapeInfo().GetShape(); // present.0.encoder.key    
    cache.encoder_kv_initialized = true;

    return next_tok;
}

// ── Subsequent decoder steps (with past KV) ───────────────────────────────────

int64_t BatchingTranslator::decoder_step_with_past(
    int64_t                    last_token,
    const std::vector<float>&  /*enc_hidden — not needed; KV already cached*/,
    const std::vector<int64_t>&/*enc_shape*/,
    const std::vector<int64_t>&enc_mask,
    KVCache&                   cache)
{
    Ort::MemoryInfo mem{"Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault};

    const int64_t enc_len      = static_cast<int64_t>(enc_mask.size());
    const int64_t past_dec_len = cache.dec_kv_shape[2];

    std::vector<int64_t> dec_ids{last_token};
    const std::vector<int64_t> dec_shape{1, 1};
    const std::vector<int64_t> enc_mask_shape{1, enc_len};

    std::vector<std::string>   in_name_strs;
    std::vector<Ort::Value>    inputs;

    // encoder_attention_mask
    in_name_strs.push_back("encoder_attention_mask");
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem,
        const_cast<int64_t*>(enc_mask.data()), enc_mask.size(),
        enc_mask_shape.data(), enc_mask_shape.size()
    ));

    // input_ids (single new token)
    in_name_strs.push_back("input_ids");
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem, dec_ids.data(), 1,
        dec_shape.data(), dec_shape.size()
    ));

    // Past KV for each layer
    for (int l = 0; l < kNumLayers; ++l) {
        // decoder past
        in_name_strs.push_back(past_name(l, "decoder", "key"));
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem,
            cache.dec_key[l].data(), cache.dec_key[l].size(),
            cache.dec_kv_shape.data(), cache.dec_kv_shape.size()
        ));
        in_name_strs.push_back(past_name(l, "decoder", "value"));
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem,
            cache.dec_val[l].data(), cache.dec_val[l].size(),
            cache.dec_kv_shape.data(), cache.dec_kv_shape.size()
        ));
        // encoder past (fixed, same every step)
        in_name_strs.push_back(past_name(l, "encoder", "key"));
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem,
            cache.enc_key[l].data(), cache.enc_key[l].size(),
            cache.enc_kv_shape.data(), cache.enc_kv_shape.size()
        ));
        in_name_strs.push_back(past_name(l, "encoder", "value"));
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem,
            cache.enc_val[l].data(), cache.enc_val[l].size(),
            cache.enc_kv_shape.data(), cache.enc_kv_shape.size()
        ));
    }

    std::vector<const char*> in_names;
    in_names.reserve(in_name_strs.size());
    for (const auto& s : in_name_strs) in_names.push_back(s.c_str());

    // Output: logits + present KV (decoder only grows; encoder KV is unchanged)
    std::vector<std::string> out_name_strs;
    out_name_strs.push_back("logits");
    for (int l = 0; l < kNumLayers; ++l) {
        out_name_strs.push_back(present_name(l, "decoder", "key"));
        out_name_strs.push_back(present_name(l, "decoder", "value"));
        // out_name_strs.push_back(present_name(l, "encoder", "key"));
        // out_name_strs.push_back(present_name(l, "encoder", "value"));
    }
    std::vector<const char*> out_names;
    out_names.reserve(out_name_strs.size());
    for (const auto& s : out_name_strs) out_names.push_back(s.c_str());

    auto outputs = dec_wp_session_->Run(
        Ort::RunOptions{nullptr},
        in_names.data(),  inputs.data(),  in_names.size(),
        out_names.data(), out_names.size()
    );

    // Argmax on logits[0, 0, :]
    const float*  logits   = outputs[0].GetTensorData<float>();
    const int64_t vocab_sz = outputs[0].GetTensorTypeAndShapeInfo().GetShape()[2];
    const int64_t next_tok = argmax(logits, vocab_sz);

    // Update decoder KV cache (append new step).
    for (int l = 0; l < kNumLayers; ++l) {
        const int base = 1 + l * 2;
        auto copy_tensor = [&](int idx) {
            const float* p = outputs[idx].GetTensorData<float>();
            const size_t n = outputs[idx].GetTensorTypeAndShapeInfo().GetElementCount();
            return std::vector<float>(p, p + n);
        };
        cache.dec_key[l] = copy_tensor(base + 0);
        cache.dec_val[l] = copy_tensor(base + 1);
    }
    // Read updated shape from the actual output tensor (source of truth):
    cache.dec_kv_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();

    return next_tok;
}

// ── Full greedy decode for one sentence ──────────────────────────────────────

std::string BatchingTranslator::translate_one(const std::string& text) 
{
    // Tokenize to int32 IDs then cast to int64.
    const auto id_batches  = tokenizer_->encode_ids_batch({text});
    const auto& ids32     = id_batches[0];
    std::cerr << "[DEBUG] input: " << text << "\n";
    std::cerr << "[DEBUG] token count: " << ids32.size() << "\n";
    std::cerr << "[DEBUG] ids: ";
    for (auto id : ids32) std::cerr << id << " ";
    std::cerr << "\n";
    std::cerr << "[DEBUG] decoder_start_token: " << cfg_.decoder_start_token << "\n";


    std::vector<int64_t> input_ids(ids32.begin(), ids32.end());
    // input_ids.push_back(0);


    if (input_ids.empty()) return "";

    // Encode.
    std::vector<int64_t> enc_mask, enc_shape;
    auto enc_hidden = run_encoder(input_ids, enc_mask, enc_shape);

    // Greedy decode with KV cache.
    KVCache cache;
    std::vector<int64_t> generated;
    generated.reserve(cfg_.max_decoding_length);

    // First step — initialises KV cache.
    int64_t next_tok = decoder_first_step(cfg_.decoder_start_token, enc_hidden, enc_shape, enc_mask, cache);
    std::cerr << "[DEBUG] first token out: " << next_tok << "\n";

    for (size_t step = 0; step < cfg_.max_decoding_length; ++step) {
        if (next_tok == cfg_.eos_token_id) break;
        if (next_tok == cfg_.pad_token_id) break;

        generated.push_back(next_tok);

        next_tok = decoder_step_with_past(
            next_tok, enc_hidden, enc_shape, enc_mask, cache
        );
    }

    std::cerr << "[DEBUG] generated ids (" << generated.size() << "): ";
    for (size_t i = 0; i < std::min(generated.size(), size_t(10)); ++i)
        std::cerr << generated[i] << " ";
    std::cerr << "\n";

    if (generated.empty()) return "";

    // 4. Decode token IDs back to string.
    std::vector<std::vector<int32_t>> id_result;
    id_result.emplace_back(generated.begin(), generated.end());
    auto decoded = tokenizer_->decode_ids_batch(id_result);
    return decoded.empty() ? "" : decoded[0];
}

// ── Batch runner ──────────────────────────────────────────────────────────────
// Called by dispatcher_loop — kept for future batched-encoder path.
// Currently each dispatcher thread calls translate_one per sentence.

void BatchingTranslator::run_batch(std::vector<WorkItem> batch) {
    // Intentionally not used in the multi-dispatcher design —
    // dispatcher_loop processes WorkItems directly. Kept for completeness.
    (void)batch;
}

} // namespace translation