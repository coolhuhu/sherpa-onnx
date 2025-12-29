#ifndef OFFLINE_ASR_ENGINE_SCHEDULER_H_
#define OFFLINE_ASR_ENGINE_SCHEDULER_H_

#include <memory>

#include "sherpa-onnx/csrc/engine/offline-asr-engine/error-code.h"

namespace sherpa_onnx {

class OfflineSession;

class Scheduler {
 public:
  Scheduler() = default;

  OfflineSession *CreateSession(ErrorCode &code);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // OFFLINE_ASR_ENGINE_SCHEDULER_H_
