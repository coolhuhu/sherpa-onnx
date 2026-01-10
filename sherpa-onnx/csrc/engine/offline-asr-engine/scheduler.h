#ifndef OFFLINE_ASR_ENGINE_SCHEDULER_H_
#define OFFLINE_ASR_ENGINE_SCHEDULER_H_

#include <memory>

#include "sherpa-onnx/csrc/engine/offline-asr-engine/error-code.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-asr-engine-config.h"

namespace sherpa_onnx {

class OfflineSession;

class Scheduler {
 public:
  Scheduler(const OfflineASREngineConfig &config);

  ~Scheduler();

  void Init(ErrorCode &code);

  void Start(ErrorCode &code);

  const OfflineSession *CreateSession(ErrorCode &code);

  void CloseSession(int32_t session_id);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // OFFLINE_ASR_ENGINE_SCHEDULER_H_
