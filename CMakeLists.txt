cmake_minimum_required(VERSION 3.20)
project(translation_server CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MYSQL_LIBRARIES /usr/lib/x86_64-linux-gnu/libmysqlclient.so)
set(MYSQL_INCLUDE_DIRS /usr/include/mysql)

# ── Release build flags 
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native -funroll-loops -ffast-math)
endif()

set(SRC_DIR       ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(THIRDPARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty)

# ── Dependencies 
find_package(Threads   REQUIRED)
find_package(Drogon    REQUIRED)
find_package(PkgConfig REQUIRED)

# ONNX Runtime
find_library(ONNXRUNTIME_LIB onnxruntime PATHS /usr/local/lib REQUIRED)
find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h PATHS /usr/local/include REQUIRED)

# SentencePiece via pkg-config (system install).
pkg_check_modules(SentencePiece REQUIRED sentencepiece)

# ── 3rd-party subdirs 
add_subdirectory(${THIRDPARTY_DIR}/pugixml EXCLUDE_FROM_ALL)

# nlohmann/json — header-only, just need the include path.
set(JSON_INCLUDE_DIR ${THIRDPARTY_DIR}/json/include)
if(NOT EXISTS ${JSON_INCLUDE_DIR}/nlohmann/json.hpp)
    message(FATAL_ERROR "nlohmann/json not found at ${JSON_INCLUDE_DIR}")
endif()

# ── Target 
add_executable(translation_server
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/translator.cpp
    ${SRC_DIR}/tokenizer.cpp
)

target_include_directories(translation_server PRIVATE
    ${SRC_DIR}
    ${JSON_INCLUDE_DIR}
    ${SentencePiece_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE}
)

target_link_libraries(translation_server PRIVATE
    Drogon::Drogon
    ${ONNXRUNTIME_LIB}
    ${SentencePiece_LIBRARIES}
    pugixml
    Threads::Threads
)

# Link against static GCC/C++ runtime so the binary is portable across
# distros without needing the exact libstdc++ version.
target_link_options(translation_server PRIVATE
    -static-libgcc
    -static-libstdc++
)