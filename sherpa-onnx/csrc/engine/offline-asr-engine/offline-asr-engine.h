#ifndef OFFLINE_ASR_ENGINE_H_
#define OFFLINE_ASR_ENGINE_H_

#include <memory>

#include "sherpa-onnx/csrc/engine/offline-asr-engine/error-code.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-asr-engine-config.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-session.h"

namespace sherpa_onnx {

class OfflineASREngineImpl;

class OfflineASREngine {
 public:
  OfflineASREngine(const OfflineASREngineConfig &config);

  void Init(ErrorCode &code);

  void Start(ErrorCode &code);

  void Shutdown();

  OfflineSession *CreateSession(ErrorCode &code);

 private:
  std::unique_ptr<OfflineASREngineImpl> impl_;
};

}  // namespace sherpa_onnx

#endif  // OFFLINE_ASR_ENGINE_H_