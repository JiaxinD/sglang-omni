// SPDX-License-Identifier: Apache-2.0
// Silero VAD over the native server, for Voxt's two detectors:
//
//   POST /v1/vad/speech_timestamps   multipart: file (16 kHz WAV), optional
//       threshold, min_speech_duration_ms, min_silence_duration_ms,
//       speech_pad_ms -> {"sample_rate", "timestamps": [{"start","end"}]}
//       in samples (Swift getSpeechTimestamps).
//   WS /v1/vad/stream   one socket per audio stream. Binary messages carry
//       16 kHz float32 little-endian samples; each whole 512-sample chunk
//       advances the stream's state, and every message is answered with
//       {"probability": p} for its last chunk, or null when none completed
//       (Swift ASRSileroStreamingVoiceActivityDetector.probability).
#pragma once

#include <filesystem>
#include <map>
#include <mutex>
#include <string>

#include "civetweb.h"
#include "silero_vad.h"

namespace silero_vad {

class VADService {
public:
  explicit VADService(const std::filesystem::path &model_directory);

  void Register(mg_context *context);
  // Request counts for /health.
  std::map<std::string, int> RequestStates() const;

  // Speech probability of each whole chunk in samples, in order.
  std::vector<float> FeedSamples(const std::vector<float> &samples,
                                 StreamState &state);
  std::vector<Timestamp> SpeechTimestamps(const std::vector<float> &samples,
                                          const TimestampOptions &options);
  const TimestampOptions &defaults() const { return vad_.defaults(); }

  void StreamOpened();
  void StreamClosed();

private:
  SileroVAD vad_;
  // One model, many callers: inference runs one call at a time.
  std::mutex inference_mutex_;
  mutable std::mutex states_mutex_;
  int open_streams_ = 0;
  int running_requests_ = 0;
};

} // namespace silero_vad
