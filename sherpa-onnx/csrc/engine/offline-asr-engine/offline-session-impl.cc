#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-session-impl.h"

#include "sherpa-onnx/csrc/engine/offline-asr-engine/scheduler.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/task.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/worker.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/offline-stream.h"

namespace sherpa_onnx {

class OfflineSessionImpl::Impl {
 public:
  Impl(int32_t session_id, Scheduler *scheduler, Worker *worker,
       OfflineSessionImpl *owner, const OfflineASREngineConfig &config)
      : start_(0),
        session_id_(session_id),
        scheduler_(scheduler),
        worker_(worker),
        owner_(owner),
        config_(config) {}

  void AcceptWaveform(int32_t sample_rate, const float *wave,
                      int32_t num_samples) {
    if (config_.use_vad) {
      VadTask task(sample_rate, start_, wave, num_samples, owner_);
      start_ += num_samples;
      worker_->CommitVadTask(std::move(task));
    } else {
      DecodeTask task(sample_rate, start_, wave, num_samples, owner_);
      start_ += num_samples;
      worker_->CommitDecodeTask(std::move(task));
    }
  }

  void Close() { scheduler_->CloseSession(session_id_); }

 private:
  int32_t start_ = 0;
  int32_t session_id_;
  Scheduler *scheduler_;
  Worker *worker_;
  OfflineSessionImpl *owner_;
  const OfflineASREngineConfig &config_;
};

OfflineSessionImpl::OfflineSessionImpl(int32_t session_id, Scheduler *scheduler,
                                       Worker *worker,
                                       const OfflineASREngineConfig &config)
    : impl_(std::make_unique<Impl>(session_id, scheduler, worker, this,
                                   config)) {}

OfflineSessionImpl::~OfflineSessionImpl() = default;

void OfflineSessionImpl::AcceptWaveform(int32_t sample_rate, const float *wave,
                                        int32_t num_samples) {
  impl_->AcceptWaveform(sample_rate, wave, num_samples);
}

void OfflineSessionImpl::Close() { impl_->Close(); }

}  // namespace sherpa_onnx
