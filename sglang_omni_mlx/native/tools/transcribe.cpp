// SPDX-License-Identifier: Apache-2.0
// Transcribes WAV files with the native runtime and prints one JSON line each,
// for parity checks against the Python reference server.
//
//   qwen3_asr_transcribe --model-path DIR [--layout voxt_swift] [--language L]
//     [--stop-at-end-of-text] [--stop-on-token-loop] [--max-new-tokens N]
//     a.wav...
//   qwen3_asr_transcribe --model-path DIR --encode TEXT
//   qwen3_asr_transcribe --model-path DIR --encode-lines FILE   (JSON string
//   per line) qwen3_asr_transcribe --dump-mel-filters OUT.f32
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "audio.h"
#include "nlohmann/json.hpp"
#include "tokenizer.h"
#include "transcriber.h"

int main(int argc, char **argv) {
  using qwen3_asr::AudioLayout;
  std::string model_path;
  std::string encode_text;
  std::string dump_mel_filters;
  std::string encode_lines;
  bool encode = false;
  qwen3_asr::TranscriptionOptions options;
  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    const auto value = [&]() { return std::string(argv[++i]); };
    if (argument == "--model-path") {
      model_path = value();
    } else if (argument == "--layout") {
      options.layout = value() == "voxt_swift" ? AudioLayout::kVoxtSwift
                                               : AudioLayout::kReference;
    } else if (argument == "--language") {
      options.language = qwen3_asr::NormalizeLanguage(value());
    } else if (argument == "--stop-at-end-of-text") {
      options.stop_at_end_of_text = true;
    } else if (argument == "--stop-on-token-loop") {
      options.stop_on_token_loop = true;
    } else if (argument == "--max-new-tokens") {
      options.max_new_tokens = std::stoi(value());
    } else if (argument == "--encode") {
      encode = true;
      encode_text = value();
    } else if (argument == "--encode-lines") {
      encode_lines = value();
    } else if (argument == "--dump-mel-filters") {
      dump_mel_filters = value();
    } else {
      files.push_back(argument);
    }
  }
  if (!dump_mel_filters.empty()) {
    std::ofstream out(dump_mel_filters, std::ios::binary);
    const auto &filters = qwen3_asr::MelFilterBank();
    const auto &window = qwen3_asr::PeriodicHannWindow();
    out.write(reinterpret_cast<const char *>(filters.data()),
              filters.size() * sizeof(float));
    out.write(reinterpret_cast<const char *>(window.data()),
              window.size() * sizeof(float));
    return 0;
  } else if (!encode_lines.empty()) {
    const qwen3_asr::Tokenizer tokenizer(model_path);
    std::ifstream lines(encode_lines);
    std::string line;
    while (std::getline(lines, line)) {
      const std::string text = nlohmann::json::parse(line).get<std::string>();
      const std::vector<int> ids = tokenizer.Encode(text);
      std::cout << nlohmann::json(
                       {{"ids", ids},
                        {"decoded", tokenizer.Decode(ids, false)},
                        {"decoded_skip", tokenizer.Decode(ids, true)}})
                       .dump()
                << "\n";
    }
    return 0;
  } else if (encode) {
    const qwen3_asr::Tokenizer tokenizer(model_path);
    std::cout << nlohmann::json(tokenizer.Encode(encode_text)).dump() << "\n";
    return 0;
  } else {
  }
  const auto load_started = std::chrono::steady_clock::now();
  const qwen3_asr::Qwen3ASRTranscriber transcriber(model_path);
  std::cerr << "loaded in "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             load_started)
                   .count()
            << " s\n";
  const std::atomic<bool> cancel(false);
  for (const auto &file : files) {
    std::ifstream stream(file, std::ios::binary);
    std::ostringstream bytes;
    bytes << stream.rdbuf();
    const auto started = std::chrono::steady_clock::now();
    const auto result = transcriber.Transcribe(
        qwen3_asr::DecodeWav(bytes.str()), options, cancel);
    nlohmann::json line = {
        {"file", file},
        {"text", result.text},
        {"language", result.language.has_value()
                         ? nlohmann::json(*result.language)
                         : nlohmann::json()},
        {"generated_token_count", result.generated_token_count},
        {"finish_reason", qwen3_asr::FinishReasonName(result.finish_reason)},
        {"seconds", std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - started)
                        .count()},
    };
    std::cout << line.dump() << std::endl;
  }
  return 0;
}
