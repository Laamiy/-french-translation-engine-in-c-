#include "translator.hpp"

#include <drogon/drogon.h>
#include <nlohmann/json.hpp>
#include <pugixml.hpp>

#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <future>

using json = nlohmann::json;

// ── Fixed-size thread pool for future completion ──────────────────────────────
// Replaces one-detached-thread-per-request with a bounded set of reusable
// threads. Under 1000 rps this pool never saturates; it only exists to avoid
// blocking Drogon's IO threads on fut.get().
class CompletionPool {
public:
    explicit CompletionPool(size_t n) {
        workers_.reserve(n);
        for (size_t i = 0; i < n; ++i)
            workers_.emplace_back([this] { run(); });
    }

    ~CompletionPool() {
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
    void run() {
        while (true) {
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

// ── Config ────────────────────────────────────────────────────────────────────

static translation::BatchingTranslator::Config load_config(const std::string& path)
{
    translation::BatchingTranslator::Config cfg;
    cfg.model_path          = "./models/onnx-en-fr-q";
    cfg.spm_dir             = "./models/onnx-en-fr-q";
    cfg.num_workers         = std::max(1u, std::thread::hardware_concurrency() / 2);
    cfg.intra_op_threads    = 2;
    cfg.max_batch_size      = 64;
    cfg.batch_timeout_us    = 5000;
    cfg.max_decoding_length = 64;

    pugi::xml_document doc;
    if (!doc.load_file(path.c_str())) {
        std::cerr << "[Warning] Cannot parse " << path << " — using defaults.\n";
        return cfg;
    }

    const auto root = doc.child("config");
    if (!root) return cfg;

    auto str = [&](const char* n, std::string& v) { if (auto c = root.child(n)) v = c.child_value(); };
    auto u64 = [&](const char* n, size_t& v)      { if (auto c = root.child(n)) v = std::stoull(c.child_value()); };
    auto u32 = [&](const char* n, uint32_t& v)    { if (auto c = root.child(n)) v = std::stoul(c.child_value()); };

    str("model_path",          cfg.model_path);
    str("spm_dir",             cfg.spm_dir);
    u64("num_workers",         cfg.num_workers);
    u64("intra_op_threads",    cfg.intra_op_threads);
    u64("max_batch_size",      cfg.max_batch_size);
    u32("batch_timeout_us",    cfg.batch_timeout_us);
    u64("max_decoding_length", cfg.max_decoding_length);

    return cfg;
}

// ── Response helpers ──────────────────────────────────────────────────────────

static drogon::HttpResponsePtr json_response(const json& body, drogon::HttpStatusCode code = drogon::k200OK)
{
    auto res = drogon::HttpResponse::newHttpResponse(code, drogon::CT_APPLICATION_JSON);
    res->setBody(body.dump());
    return res;
}

static drogon::HttpResponsePtr error_response(drogon::HttpStatusCode code, const std::string& msg)
{
    return json_response({{"error", msg}}, code);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    const std::string config_path = (argc > 1) ? argv[1] : "config.xml";
    auto cfg = load_config(config_path);

    std::cout << "=== Translation Server ===\n"
              << "  model_path        : " << cfg.model_path          << "\n"
              << "  spm_dir           : " << cfg.spm_dir             << "\n"
              << "  num_workers       : " << cfg.num_workers         << "\n"
              << "  intra_op_threads  : " << cfg.intra_op_threads    << "\n"
              << "  max_batch_size    : " << cfg.max_batch_size      << "\n"
              << "  batch_timeout_us  : " << cfg.batch_timeout_us    << " µs\n"
              << "  max_decoding_len  : " << cfg.max_decoding_length << "\n";

    auto translator = std::make_shared<translation::BatchingTranslator>(std::move(cfg));

    // Completion pool: enough threads to drain futures without blocking IO.
    // num_workers is the max number of in-flight translations at any moment,
    // so that's the ceiling on how many futures can be pending simultaneously.
    const size_t completion_threads = translator->num_workers();
    auto pool = std::make_shared<CompletionPool>(completion_threads);

    const int drogon_threads = static_cast<int>(std::min(std::thread::hardware_concurrency(), 8u));
    drogon::app().setThreadNum(drogon_threads);
    drogon::app().setClientMaxBodySize(1024 * 1024);
    drogon::app().setClientMaxMemoryBodySize(1024 * 1024);
    drogon::app().enableGzip(true);
    drogon::app().addListener("0.0.0.0", 8888);
    drogon::app().disableSigtermHandling();

    // ── POST /translate ───────────────────────────────────────────────────────
    drogon::app().registerHandler(
        "/translate",
        [translator, pool](
            const drogon::HttpRequestPtr& req,
            std::function<void(const drogon::HttpResponsePtr&)>&& callback)
        {
            json body;
            try {
                body = json::parse(req->getBody());
            } catch (const json::exception&) {
                callback(error_response(drogon::k400BadRequest, "Body is not valid JSON."));
                return;
            }

            if (!body.contains("item")) {
                callback(error_response(drogon::k400BadRequest, "Missing required field: 'item'."));
                return;
            }

            std::vector<std::string> texts;
            const auto& item = body["item"];

            if (item.is_string()) {
                texts.push_back(item.get<std::string>());
            } else if (item.is_array()) {
                texts.reserve(item.size());
                for (const auto& el : item) {
                    if (!el.is_string()) {
                        callback(error_response(drogon::k400BadRequest, "All elements of 'item' must be strings."));
                        return;
                    }
                    texts.push_back(el.get<std::string>());
                }
            } else {
                callback(error_response(drogon::k400BadRequest, "'item' must be a string or array of strings."));
                return;
            }

            if (texts.empty()) {
                callback(json_response({{"results", json::array()}}));
                return;
            }

            // Enqueue translation — non-blocking, returns future immediately.
            auto fut = std::make_shared<std::future<std::vector<std::string>>>(
                translator->translate_async(texts)
            );

            // Hand off to completion pool — no new threads spawned per request.
            pool->post([
                texts = std::move(texts),
                fut   = std::move(fut),
                cb    = std::move(callback)
            ]() mutable {
                // wait_for with timeout — prevents a stuck inference from
                // leaking a completion thread forever.
                if (fut->wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
                    cb(error_response(drogon::k503ServiceUnavailable, "Translation timeout."));
                    return;
                }
                try {
                    const auto translations = fut->get();
                    json out;
                    out["results"] = json::array();
                    for (size_t i = 0; i < texts.size(); ++i) {
                        out["results"].push_back({
                            {"source",      texts[i]},
                            {"translation", i < translations.size() ? translations[i] : ""}
                        });
                    }
                    cb(json_response(out));
                } catch (const std::exception& e) {
                    cb(error_response(drogon::k500InternalServerError, e.what()));
                }
            });
        },
        {drogon::Post}
    );

    // ── GET /health ───────────────────────────────────────────────────────────
    drogon::app().registerHandler(
        "/health",
        [](const drogon::HttpRequestPtr&, std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            callback(json_response({{"status", "ok"}}));
        },
        {drogon::Get}
    );

    std::cout << "Listening on 0.0.0.0:8888  (Drogon threads: " << drogon_threads << ")\n";
    drogon::app().run();
    return 0;
}