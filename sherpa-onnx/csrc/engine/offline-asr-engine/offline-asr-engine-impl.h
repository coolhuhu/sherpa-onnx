#ifndef OFFLINE_ASR_ENGINE_IMPL_H_
#define OFFLINE_ASR_ENGINE_IMPL_H_

#include "sherpa-onnx/csrc/engine/offline-asr-engine/error-code.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-asr-engine-config.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-session.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/scheduler.h"

namespace sherpa_onnx {

class OfflineASREngineImpl {
 public:
  OfflineASREngineImpl(const OfflineASREngineConfig &config)
      : scheduler_(config) {}

  ~OfflineASREngineImpl() = default;

  void Init(ErrorCode &code) { scheduler_.Init(code); }

  void Start(ErrorCode &code) { scheduler_.Start(code); }

  const OfflineSession *CreateSession(ErrorCode &code) {
    return scheduler_.CreateSession(code);
  }

 private:
  Scheduler scheduler_;
};

}  // namespace sherpa_onnx

#endif  // OFFLINE_ASR_ENGINE_IMPL_H_