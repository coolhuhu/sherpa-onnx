#ifndef OFFLINE_ASR_ENGINE_IMPL_H_
#define OFFLINE_ASR_ENGINE_IMPL_H_

#include "sherpa-onnx/csrc/engine/offline-asr-engine/error-code.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-asr-engine-config.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-session.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/scheduler.h"

namespace sherpa_onnx {

class OfflineASREngineImpl {
 public:
  OfflineASREngineImpl(const OfflineASREngineConfig &config) {}

  ~OfflineASREngineImpl() = default;

  OfflineSession *CreateSession(ErrorCode &code) {}

 private:
  OfflineASREngineConfig config_;
  Scheduler scheduler_;
};

}  // namespace sherpa_onnx

#endif  // OFFLINE_ASR_ENGINE_IMPL_H_