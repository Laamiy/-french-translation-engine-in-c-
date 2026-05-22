#include "translator.hpp"

#include <drogon/drogon.h>
#include <nlohmann/json.hpp>
#include <pugixml.hpp>

#include <iostream>
#include <string>
#include <thread>

using json = nlohmann::json;


static translation::BatchingTranslator::Config load_config(const std::string& path) 
{
    translation::BatchingTranslator::Config cfg;
    // Defaults.
    cfg.model_path          = "./models/clean-en-fr-ct2-f32";
    cfg.spm_dir             = "./models/clean-en-fr";
    cfg.num_workers         = std::max(1u, std::thread::hardware_concurrency() / 2);
    cfg.intra_op_threads    = 2;
    cfg.max_batch_size      = 64;
    cfg.batch_timeout_us    = 5000;
    cfg.beam_size           = 1;
    cfg.max_decoding_length = 128;

    pugi::xml_document doc;

    if (!doc.load_file(path.c_str())) 
    {
        std::cerr << "[Warning] Cannot parse " << path << " — using defaults.\n";
        return cfg;
    }

    const auto root = doc.child("config");
    if (!root) 
     return cfg;

    auto str  = [&](const char* n, std::string& v)  { if (auto c = root.child(n)) v = c.child_value(); };
    auto u64  = [&](const char* n, size_t& v)        { if (auto c = root.child(n)) v = std::stoull(c.child_value()); };
    auto u32  = [&](const char* n, uint32_t& v)      { if (auto c = root.child(n)) v = std::stoul(c.child_value()); };

    str("model_path",          cfg.model_path);
    str("spm_dir",             cfg.spm_dir);
    u64("num_workers",         cfg.num_workers);
    u64("intra_op_threads",    cfg.intra_op_threads);
    u64("max_batch_size",      cfg.max_batch_size);
    u32("batch_timeout_us",    cfg.batch_timeout_us);
    u64("beam_size",           cfg.beam_size);
    u64("max_decoding_length", cfg.max_decoding_length);

    return cfg;
}


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
              << "  beam_size         : " << cfg.beam_size           << "\n"
              << "  max_decoding_len  : " << cfg.max_decoding_length << "\n";

    // Construct before Drogon starts — this loads the model and warms CT2.
    auto translator = std::make_shared<translation::BatchingTranslator>(std::move(cfg));

    // Drogon: one IO thread per hardware core is overkill for a proxy-style
    // server; 4–8 is the sweet spot.  All the real work is in CT2's pool.
    const int drogon_threads = static_cast<int>( std::min(std::thread::hardware_concurrency(), 8u));
    drogon::app().setThreadNum(drogon_threads);
    drogon::app().addListener("0.0.0.0", 8888);
    drogon::app().disableSigtermHandling(); // OS terminate 

    // ── POST /translate 
    // Accepts:
    //   { "item": "single string" }
    //   { "item": ["string1", "string2", ...] }
    // Returns:
    //   { "results": [{ "source": "...", "translation": "..." }, ...] }
    drogon::app().registerHandler(
        "/translate",
        [translator](const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& cb)
        {
            json body;
            try 
            {
                body = json::parse(req->getBody());
            } 
            catch (const json::exception&) 
            {
                cb(error_response(drogon::k400BadRequest, "Body is not valid JSON."));
                return;
            }

            if (!body.contains("item")) 
            {
                cb(error_response(drogon::k400BadRequest, "Missing required field: 'item'."));
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
                    if (!el.is_string()) {
                        cb(error_response(drogon::k400BadRequest, "All elements of 'item' must be strings."));
                        return;
                    }
                    texts.push_back(el.get<std::string>());
                }
            } 
            else 
            {
                cb(error_response(drogon::k400BadRequest, "'item' must be a string or array of strings."));
                return;
            }

            if (texts.empty()) 
            {
                cb(json_response({{"results", json::array()}}));
                return;
            }

            // Enqueue — returns immediately without blocking the IO thread.
            auto fut = translator->translate_async(texts);

            // Move the future and callback into a detached wait thread.
            // This thread does nothing but block on the future; it costs one
            // kernel thread but keeps Drogon's IO threads free.
            // Alternative: use a polling scheme on a timer thread, but this
            // is simpler and perfectly safe — the future is always resolved.
            std::thread( [
                        texts     = std::move(texts),
                        fut       = std::move(fut),
                        cb        = std::move(cb)
                    ]() mutable 
                    {
                        try {
                            const auto translations = fut.get(); // blocks this thread only

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
                    }).detach();
        },
                {drogon::Post}
    );

    // ── GET /health 
    drogon::app().registerHandler( "/health", [](const drogon::HttpRequestPtr&, std::function<void(const drogon::HttpResponsePtr&)>&& cb) 
                        {
                            cb(json_response({{"status", "ok"}}));
                        }, {drogon::Get}
                    );

    std::cout << "Listening on 0.0.0.0:8888  (Drogon threads: " << drogon_threads << ")\n";
    drogon::app().run();
    return 0;
}