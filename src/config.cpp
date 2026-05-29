#include "config.hpp"



translation::BatchingTranslator::Config utils::load_config(const std::string& path)
{
    translation::BatchingTranslator::Config cfg;


    cfg.model_path          = "./models/onnx-en-fr-q";
    cfg.spm_dir             = "./models/onnx-en-fr-q";
    cfg.num_workers         = std::max(1u, std::thread::hardware_concurrency() / 2);
    cfg.intra_op_threads    = 1;
    cfg.max_batch_size      = 64;
    cfg.batch_timeout_us    = 5000;
    cfg.max_decoding_length = 64;

    pugi::xml_document doc;
    pugi::xml_parse_result res = doc.load_file(path.c_str());
    
    if (!res) 
    {
        std::cerr << "[Warning] Cannot parse " << path <<'\n' ; 
        std::cerr << res.description() << " — using defaults.\n";
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

    char* num_workers = std::getenv("NWORKERS");
    cfg.num_workers   = num_workers ? std::stoi(num_workers) : cfg.num_workers ;  

    return cfg;
}


drogon::HttpResponsePtr utils::json_response(const nlohmann::json& body, drogon::HttpStatusCode code )
{
    drogon::HttpResponsePtr res = drogon::HttpResponse::newHttpResponse(code, drogon::CT_APPLICATION_JSON);
    res->setBody(body.dump());
    return res;
}

drogon::HttpResponsePtr utils::error_response(drogon::HttpStatusCode code, const std::string& msg)
{
    return json_response({{"error", msg}}, code);
}
