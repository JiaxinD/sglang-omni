# SPDX-License-Identifier: Apache-2.0
# Native Qwen3-ASR runtime on MLX: one binary, no Python.
#
#   cmake -S sglang_omni_mlx/native -B build/native -G Ninja \
#     -DCMAKE_BUILD_TYPE=Release -DMLX_DIR=<mlx>/share/cmake/MLX
#   cmake --build build/native
cmake_minimum_required(VERSION 3.24)
project(sglang_omni_mlx_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The same MLX build the Python reference server runs on, so both produce
# bit-identical outputs.
find_package(MLX CONFIG REQUIRED)

# civetweb and PCRE2 still declare pre-3.5 minimums.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
set(PCRE2_BUILD_PCRE2GREP OFF CACHE BOOL "" FORCE)
set(PCRE2_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PCRE2_SUPPORT_UNICODE ON CACHE BOOL "" FORCE)
set(PCRE2_SUPPORT_JIT OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  pcre2
  GIT_REPOSITORY https://github.com/PCRE2Project/pcre2.git
  GIT_TAG pcre2-10.44
  GIT_SHALLOW TRUE)
set(CIVETWEB_ENABLE_WEBSOCKETS ON CACHE BOOL "" FORCE)
set(CIVETWEB_ENABLE_SSL OFF CACHE BOOL "" FORCE)
set(CIVETWEB_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(CIVETWEB_ENABLE_SERVER_EXECUTABLE OFF CACHE BOOL "" FORCE)
set(CIVETWEB_ENABLE_CXX OFF CACHE BOOL "" FORCE)
set(CIVETWEB_ENABLE_ASAN OFF CACHE BOOL "" FORCE)
set(CIVETWEB_INSTALL_EXECUTABLE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  civetweb
  GIT_REPOSITORY https://github.com/civetweb/civetweb.git
  GIT_TAG v1.16
  GIT_SHALLOW TRUE)
set(BUILD_SHARED_LIBS OFF)
FetchContent_MakeAvailable(nlohmann_json pcre2 civetweb)

add_library(qwen3_asr STATIC
  src/audio.cpp
  src/model.cpp
  src/tokenizer.cpp
  src/transcriber.cpp)
target_include_directories(qwen3_asr PUBLIC src)
target_link_libraries(qwen3_asr PUBLIC mlx nlohmann_json::nlohmann_json pcre2-8-static
  "-framework CoreFoundation")
target_compile_options(qwen3_asr PUBLIC ${MLX_CXX_FLAGS} PRIVATE -Wall -Wextra)

# Parity tool: transcribes WAV files and prints one JSON line per file.
add_executable(qwen3_asr_transcribe tools/transcribe.cpp)
target_link_libraries(qwen3_asr_transcribe PRIVATE qwen3_asr)

set_target_properties(qwen3_asr_transcribe PROPERTIES
  BUILD_RPATH "${MLX_LIBRARY_DIRS};@loader_path"
  INSTALL_RPATH "@loader_path")
