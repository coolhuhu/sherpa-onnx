#ifndef SHERPA_ONNX_OFFLINE_ENGINE_VAD_H_
#define SHERPA_ONNX_OFFLINE_ENGINE_VAD_H_

#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/vad-model-config.h"

namespace sherpa_onnx {

struct VadSpeechSegment {
  int32_t start;  // in samples
  std::vector<float> samples;
  bool endpoint;  // this is an empty packet if endpoint = true and start = -1

  VadSpeechSegment() : start(-1), endpoint(false) {}
};

class OnlineVoiceActivityDetector {
 public:
  OnlineVoiceActivityDetector(const VadModelConfig &config);

  ~OnlineVoiceActivityDetector();

  void AcceptWaveform(const float *samples, int32_t n);
  void AcceptWaveform(const std::vector<float> &samples);

  bool Empty() const;

  void PopAll();

  void Flush() const;

  std::vector<VadSpeechSegment> GetSpeechSegment();

  void Reset();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_OFFLINE_ENGINE_VAD_H_