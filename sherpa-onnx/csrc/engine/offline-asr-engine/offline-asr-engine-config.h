#ifndef OFFLINE_ASR_ENGINE_CONFIG_H_
#define OFFLINE_ASR_ENGINE_CONFIG_H_

#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/vad-model-config.h"

namespace sherpa_onnx {

struct OfflineASREngineConfig {
  int32_t num_workers = 1;
  bool enable_gpu = false;
  bool use_vad = false;
  bool enable_task_stealing = false;

  VadModelConfig vad_config;
  OfflineRecognizerConfig recognizer_config;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // OFFLINE_ASR_ENGINE_CONFIG_H_