#pragma once 
#include <iostream>
#include <pugixml.hpp>
#include "translator.hpp"
#include <drogon/HttpResponse.h>
#include <drogon/HttpResponse.h>
#include <drogon/HttpTypes.h>
#include <drogon/drogon.h>
#include <nlohmann/json.hpp>



// Response helpers :
namespace utils 
{
    translation::BatchingTranslator::Config load_config(const std::string& path);
    drogon::HttpResponsePtr json_response(const nlohmann::json& body, drogon::HttpStatusCode code = drogon::k200OK);
    drogon::HttpResponsePtr error_response(drogon::HttpStatusCode code, const std::string& msg);
}// translation
