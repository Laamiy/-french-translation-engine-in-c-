#include "translator.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace translation {

namespace fs = std::filesystem;

// ── Name helpers — computed once at file scope ────────────────────────────────

static std::string present_name(int l, const char* side, const char* kv) 
{
    return "present." + std::to_string(l) + "." + side + "." + kv;
}
static std::string past_name(int l, const char* side, const char* kv) 
{
    return "past_key_values." + std::to_string(l) + "." + side + "." + kv;
}
static int64_t argmax(const float* data, int64_t n) 
{
    return static_cast<int64_t>(std::max_element(data, data + n) - data);
}
// ── Construction ──────────────────────────────────────────────────────────────

BatchingTranslator::BatchingTranslator(Config cfg)
    : cfg_(std::move(cfg))
    , ort_env_(ORT_LOGGING_LEVEL_WARNING, "translation_server")
    , mem_info_("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault)
{
    tokenizer_ = std::make_unique<Tokenizer>(cfg_.spm_dir);

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

    // ── Pre-build all I/O name arrays once ───────────────────────────────────

    dec_first_out_strs_.push_back("logits");
    for (int l = 0; l < kNumLayers; ++l) 
    {
        dec_first_out_strs_.push_back(present_name(l, "decoder", "key"));
        dec_first_out_strs_.push_back(present_name(l, "decoder", "value"));
        dec_first_out_strs_.push_back(present_name(l, "encoder", "key"));
        dec_first_out_strs_.push_back(present_name(l, "encoder", "value"));
    }
    for (const auto& s : dec_first_out_strs_)
        dec_first_out_names_.push_back(s.c_str());

    dec_wp_in_strs_.push_back("encoder_attention_mask");
    dec_wp_in_strs_.push_back("input_ids");
    for (int l = 0; l < kNumLayers; ++l) {
        dec_wp_in_strs_.push_back(past_name(l, "decoder", "key"));
        dec_wp_in_strs_.push_back(past_name(l, "decoder", "value"));
        dec_wp_in_strs_.push_back(past_name(l, "encoder", "key"));
        dec_wp_in_strs_.push_back(past_name(l, "encoder", "value"));
    }
    for (const auto& s : dec_wp_in_strs_)
        dec_wp_in_names_.push_back(s.c_str());

    dec_wp_out_strs_.push_back("logits");
    for (int l = 0; l < kNumLayers; ++l) {
        dec_wp_out_strs_.push_back(present_name(l, "decoder", "key"));
        dec_wp_out_strs_.push_back(present_name(l, "decoder", "value"));
    }
    for (const auto& s : dec_wp_out_strs_)
        dec_wp_out_names_.push_back(s.c_str());

    // ── Spawn dispatchers first, then warmup ──────────────────────────────────
    dispatchers_.reserve(cfg_.num_workers);
    for (size_t i = 0; i < cfg_.num_workers; ++i)
        dispatchers_.emplace_back(&BatchingTranslator::dispatcher_loop, this);

    std::cout << "Warming up...\n";
    std::vector<std::future<std::vector<std::string>>> wf;
    wf.reserve(cfg_.num_workers);
    for (size_t i = 0; i < cfg_.num_workers; ++i)
        wf.push_back(translate_async({"warm up"}));
    for (auto& f : wf) f.get();
    std::cout << "Warmup complete.\n";
}

BatchingTranslator::~BatchingTranslator() {
    stop_.store(true, std::memory_order_release);
    queue_cv_.notify_all();
    for (auto& t : dispatchers_)
        if (t.joinable()) t.join();
}

// ── Public API ────────────────────────────────────────────────────────────────

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

// ── Dispatcher ────────────────────────────────────────────────────────────────

void BatchingTranslator::dispatcher_loop() {
    // Per-thread reusable KVCache — avoids heap allocation on every request.
    // Each dispatcher thread owns one; never shared.
    KVCache cache;
    cache.dec_key.resize(kNumLayers);
    cache.dec_val.resize(kNumLayers);
    cache.enc_key.resize(kNumLayers);
    cache.enc_val.resize(kNumLayers);

    while (true) {
        WorkItem item;
        {
            std::unique_lock lock{queue_mtx_};
            queue_cv_.wait(lock, [this] {
                return !queue_.empty() || stop_.load(std::memory_order_acquire);
            });
            if (stop_.load(std::memory_order_acquire) && queue_.empty()) return;
            item = std::move(queue_.front());
            queue_.pop_front();
        }

        try {
            std::vector<std::string> results;
            results.reserve(item.texts.size());
            for (const auto& text : item.texts)
                results.push_back(translate_one(text, cache));
            item.promise.set_value(std::move(results));
        } catch (...) {
            item.promise.set_exception(std::current_exception());
        }
    }
}

// ── Encoder ───────────────────────────────────────────────────────────────────

std::vector<float> BatchingTranslator::run_encoder(
    const std::vector<int64_t>& input_ids,
    std::vector<int64_t>&       out_mask,
    std::vector<int64_t>&       out_enc_shape)
{
    const int64_t seq = static_cast<int64_t>(input_ids.size());
    const std::vector<int64_t> shape{1, seq};
    out_mask.assign(seq, 1LL);

    std::array<Ort::Value, 2> inputs = 
    {
        Ort::Value::CreateTensor<int64_t>(mem_info_,
            const_cast<int64_t*>(input_ids.data()), input_ids.size(),
            shape.data(), shape.size()),

        Ort::Value::CreateTensor<int64_t>(mem_info_,
            out_mask.data(), out_mask.size(),
            shape.data(), shape.size())
    };

    static const char* in_names[]  = {"input_ids", "attention_mask"};
    static const char* out_names[] = {"last_hidden_state"};

    auto outputs = enc_session_->Run(
        Ort::RunOptions{nullptr},
        in_names, inputs.data(), 2,
        out_names, 1);

    out_enc_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const float* ptr = outputs[0].GetTensorData<float>();
    const size_t n   = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    return {ptr, ptr + n};
}

// ── First decoder step ────────────────────────────────────────────────────────

int64_t BatchingTranslator::decoder_first_step(
    int64_t                     start_token,
    const std::vector<float>&   enc_hidden,
    const std::vector<int64_t>& enc_shape,
    const std::vector<int64_t>& enc_mask,
    KVCache&                    cache)
{
    const int64_t enc_len = enc_shape[1];
    int64_t dec_id = start_token;
    const std::vector<int64_t> dec_shape{1, 1};
    const std::vector<int64_t> enc_mask_shape{1, enc_len};

    static const char* in_names[] = {
        "encoder_attention_mask", "input_ids", "encoder_hidden_states"
    };

    std::array<Ort::Value, 3> inputs = {
        Ort::Value::CreateTensor<int64_t>(mem_info_,
            const_cast<int64_t*>(enc_mask.data()), enc_mask.size(),
            enc_mask_shape.data(), enc_mask_shape.size()),
        Ort::Value::CreateTensor<int64_t>(mem_info_,
            &dec_id, 1,
            dec_shape.data(), dec_shape.size()),
        Ort::Value::CreateTensor<float>(mem_info_,
            const_cast<float*>(enc_hidden.data()), enc_hidden.size(),
            enc_shape.data(), enc_shape.size())
    };

    auto outputs = dec_session_->Run(
        Ort::RunOptions{nullptr},
        in_names, inputs.data(), 3,
        dec_first_out_names_.data(), dec_first_out_names_.size());

    const float*  logits   = outputs[0].GetTensorData<float>();
    const int64_t vocab_sz = outputs[0].GetTensorTypeAndShapeInfo().GetShape()[2];
    const int64_t next_tok = argmax(logits, vocab_sz);

    // Copy KV tensors into the per-thread cache.
    // Output layout: logits, [dec_key, dec_val, enc_key, enc_val] * kNumLayers
    for (int l = 0; l < kNumLayers; ++l) {
        const int base = 1 + l * 4;
        // auto copy = [&](int idx) -> std::vector<float>& { return *reinterpret_cast<std::vector<float>*>(nullptr); };
        // Inline copy to avoid lambda overhead:
        auto do_copy = [&](int idx, std::vector<float>& dst) {
            const float* p = outputs[idx].GetTensorData<float>();
            const size_t n = outputs[idx].GetTensorTypeAndShapeInfo().GetElementCount();
            dst.assign(p, p + n);
        };
        do_copy(base + 0, cache.dec_key[l]);
        do_copy(base + 1, cache.dec_val[l]);
        do_copy(base + 2, cache.enc_key[l]);
        do_copy(base + 3, cache.enc_val[l]);
    }

    cache.dec_kv_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    cache.enc_kv_shape = outputs[3].GetTensorTypeAndShapeInfo().GetShape();
    cache.encoder_kv_initialized = true;

    return next_tok;
}

// ── Subsequent decoder steps ──────────────────────────────────────────────────

int64_t BatchingTranslator::decoder_step_with_past(
    int64_t                     last_token,
    const std::vector<int64_t>& enc_mask,
    KVCache&                    cache)
{
    const int64_t enc_len = static_cast<int64_t>(enc_mask.size());
    int64_t dec_id = last_token;
    const std::vector<int64_t> dec_shape{1, 1};
    const std::vector<int64_t> enc_mask_shape{1, enc_len};

    std::vector<Ort::Value> inputs;
    inputs.reserve(2 + kNumLayers * 4);

    inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem_info_,
        const_cast<int64_t*>(enc_mask.data()), enc_mask.size(),
        enc_mask_shape.data(), enc_mask_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem_info_,
        &dec_id, 1,
        dec_shape.data(), dec_shape.size()));

    for (int l = 0; l < kNumLayers; ++l) {
        inputs.push_back(Ort::Value::CreateTensor<float>(mem_info_,
            cache.dec_key[l].data(), cache.dec_key[l].size(),
            cache.dec_kv_shape.data(), cache.dec_kv_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(mem_info_,
            cache.dec_val[l].data(), cache.dec_val[l].size(),
            cache.dec_kv_shape.data(), cache.dec_kv_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(mem_info_,
            cache.enc_key[l].data(), cache.enc_key[l].size(),
            cache.enc_kv_shape.data(), cache.enc_kv_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(mem_info_,
            cache.enc_val[l].data(), cache.enc_val[l].size(),
            cache.enc_kv_shape.data(), cache.enc_kv_shape.size()));
    }

    auto outputs = dec_wp_session_->Run(
        Ort::RunOptions{nullptr},
        dec_wp_in_names_.data(), inputs.data(), dec_wp_in_names_.size(),
        dec_wp_out_names_.data(), dec_wp_out_names_.size());

    const float*  logits   = outputs[0].GetTensorData<float>();
    const int64_t vocab_sz = outputs[0].GetTensorTypeAndShapeInfo().GetShape()[2];
    const int64_t next_tok = argmax(logits, vocab_sz);

    // Reuse existing cache buffers — assign() keeps capacity, avoids realloc.
    for (int l = 0; l < kNumLayers; ++l) {
        const int base = 1 + l * 2;
        auto do_copy = [&](int idx, std::vector<float>& dst) {
            const float* p = outputs[idx].GetTensorData<float>();
            const size_t n = outputs[idx].GetTensorTypeAndShapeInfo().GetElementCount();
            dst.assign(p, p + n);
        };
        do_copy(base,     cache.dec_key[l]);
        do_copy(base + 1, cache.dec_val[l]);
    }
    cache.dec_kv_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();

    return next_tok;
}

// ── Full greedy decode ────────────────────────────────────────────────────────

std::string BatchingTranslator::translate_one(const std::string& text, KVCache& cache) {
    const auto id_batches = tokenizer_->encode_ids_batch({text});
    const auto& ids32     = id_batches[0];

    std::vector<int64_t> input_ids(ids32.begin(), ids32.end());
    if (input_ids.empty()) return "";

    std::vector<int64_t> enc_mask, enc_shape;
    auto enc_hidden = run_encoder(input_ids, enc_mask, enc_shape);

    std::vector<int64_t> generated;
    generated.reserve(cfg_.max_decoding_length);

    int64_t next_tok = decoder_first_step(
        cfg_.decoder_start_token, enc_hidden, enc_shape, enc_mask, cache);

    for (size_t step = 0; step < cfg_.max_decoding_length; ++step) {
        if (next_tok == cfg_.eos_token_id) break;
        if (next_tok == cfg_.pad_token_id) break;
        generated.push_back(next_tok);
        next_tok = decoder_step_with_past(next_tok, enc_mask, cache);
    }

    if (generated.empty()) return "";

    std::vector<std::vector<int32_t>> id_result;
    id_result.emplace_back(generated.begin(), generated.end());
    auto decoded = tokenizer_->decode_ids_batch(id_result);
    return decoded.empty() ? "" : decoded[0];
}

void BatchingTranslator::run_batch(std::vector<WorkItem> batch) { (void)batch; }

} // namespace translation