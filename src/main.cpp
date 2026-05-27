
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <future>
#include <functional>
#include <condition_variable>

#include <drogon/drogon.h>
#include <nlohmann/json.hpp>
#include <pugixml.hpp>

#include "config.hpp"
#include "translator.hpp"
#include "CompletionPool.hpp"
using json = nlohmann::json;

// ── Fixed-size thread pool for future completion 
// Replaces one-detached-thread-per-request with a bounded set of reusable
// threads. Under 1000 rps this pool never saturates; it only exists to avoid
// blocking Drogon's IO threads on fut.get().


typedef std::function<void(const drogon::HttpResponsePtr&)> HttpResponsePtr; 

int main(int argc, char* argv[])
{
    const std::string config_path = (argc > 1) ? argv[1] : "config.xml";
    auto cfg = utils::load_config(config_path);

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

    auto pool = std::make_shared<translation::CompletionPool>(completion_threads);

    const int drogon_threads = static_cast<int>(std::min(std::thread::hardware_concurrency(), 8u));
    drogon::app().setThreadNum(drogon_threads);
    drogon::app().setClientMaxBodySize(1024 * 1024);
    drogon::app().setClientMaxMemoryBodySize(1024 * 1024);
    drogon::app().enableGzip(true);

    drogon::app().addListener("0.0.0.0", 8888);
    drogon::app().disableSigtermHandling();

    // POST /translate 
    drogon::app().registerHandler( "/translate", [translator, pool] ( const drogon::HttpRequestPtr& req, HttpResponsePtr&& callback)
        {
            json body;
            try 
            {
                body = json::parse(req->getBody());
            } 
            catch (const json::exception&) 
            {
                callback(utils::error_response(drogon::k400BadRequest, "Body is not valid JSON."));
                return;
            }

            if (!body.contains("item")) 
            {
                callback(utils::error_response(drogon::k400BadRequest, "Missing required field: 'item'."));
                return;
            }

            std::vector<std::string> texts;
            const auto& item = body["item"];

            if (item.is_string()) 
            {
                texts.push_back(item.get<std::string>());
            } 
            else if (item.is_array()) 
            {
                texts.reserve(item.size());
                for (const auto& el : item) 
                {
                    if (!el.is_string()) 
                    {
                        callback(utils::error_response(drogon::k400BadRequest, "All elements of 'item' must be strings."));
                        return;
                    }
                    texts.push_back(el.get<std::string>());
                }
            } 
            else 
            {
                callback(utils::error_response(drogon::k400BadRequest, "'item' must be a string or array of strings."));
                return;
            }

            if (texts.empty()) 
            {
                callback(utils::json_response({{"results", json::array()}}));
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
                    cb(utils::error_response(drogon::k503ServiceUnavailable, "Translation timeout."));
                    return;
                }
                try 
                {
                    const auto translations = fut->get();
                    json out;
                    out["results"] = json::array();
                    for (size_t i = 0; i < texts.size(); ++i) 
                    {
                        out["results"].push_back({
                            {"source",      texts[i]},
                            {"translation", i < translations.size() ? translations[i] : ""}
                        });
                    }
                    cb(utils::json_response(out));
                } 
                catch (const std::exception& e) 
                {
                    cb(utils::error_response(drogon::k500InternalServerError, e.what()));
                }
            });
        },
        {drogon::Post}
    );

    // GET health 
    drogon::app().registerHandler( "/health", [](const drogon::HttpRequestPtr&, HttpResponsePtr&& callback) 
        {
            callback(utils::json_response({{"status", "ok"}}));
        },
        {drogon::Get}
    );
    std::cout << "Listening on 0.0.0.0:8888  (Drogon threads: " << drogon_threads << ")\n";
    
    drogon::app().run();

    return 0;
}