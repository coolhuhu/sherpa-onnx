#ifndef OFFLINE_ASR_ENGINE_SESSION_IMPL_H_
#define OFFLINE_ASR_ENGINE_SESSION_IMPL_H_

#include <memory>

#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-asr-engine-config.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-session.h"

namespace sherpa_onnx {

class Worker;
class Scheduler;

class OfflineSessionImpl : public OfflineSession {
 public:
  OfflineSessionImpl(int32_t session_id, Scheduler *scheduler, Worker *worker,
                     const OfflineASREngineConfig &config);

  ~OfflineSessionImpl();

  void AcceptWaveform(int32_t sample_rate, const float *wave,
                      int32_t num_samples) override;

  void Close() override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // OFFLINE_ASR_ENGINE_SESSION_IMPL_H_