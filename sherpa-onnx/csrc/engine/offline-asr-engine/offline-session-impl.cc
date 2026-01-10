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
       OfflineSessionImpl *owner, OnlineVoiceActivityDetector *vad_detector,
       const OfflineASREngineConfig &config)
      : config_(config),
        start_(0),
        session_id_(session_id),
        scheduler_(scheduler),
        worker_(worker),
        vad_detector_(vad_detector),
        owner_(owner),
        is_decode_finished_(false),
        is_input_finished_(false) {}

  void AcceptWaveform(int32_t sample_rate, const float *wave,
                      int32_t num_samples, ErrorCode &error_code) {
    if (num_samples <= 0 || num_samples > config_.max_accept_waveform_size) {
      error_code.error_code = ErrorCode::kInvalidArgument;
      error_code.error_msg =
          "Invalid argument in AcceptWaveform: num_samples <= 0 ||  "
          "num_samples > config_.max_accept_waveform_size";
      return;
    }

    if (!config_.enable_vad &&
        num_samples + start_ > config_.max_model_input_samples) {
      error_code.error_code = ErrorCode::kInvalidArgument;
      error_code.error_msg =
          "Invalid argument in AcceptWaveform: input wave is too long";
      return;
    }

    WaveTask task(sample_rate, start_, wave, num_samples, owner_);
    start_ += num_samples;
    worker_->CommitWaveTask(std::move(task));
  }

  void Close() { scheduler_->CloseSession(session_id_); }

  void InputFinished() { is_input_finished_ = true; }

  bool IsInputFinished() const { return is_input_finished_; }

  // 获取当前识别结果（聚合所有已完成的片段）
  std::vector<OfflineRecognitionResult> GetResults() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::sort(results_.begin(), results_.end(),
              [](const auto &r1, const auto &r2) {
                return r1.segment_id < r2.segment_id;
              });
    return results_;
  }

  void AddResult(const OfflineRecognitionResult &result) {
    std::lock_guard<std::mutex> lock(mutex_);
    results_.push_back(result);
  }
  void AddResult(OfflineRecognitionResult &&result) {
    std::lock_guard<std::mutex> lock(mutex_);
    results_.push_back(std::move(result));
  }

  bool IsDecodeFinished() const { return is_decode_finished_; }

  void DecodeFinished() { is_decode_finished_ = true; }

  int32_t SessionID() const { return session_id_; }

  int32_t WorkerID() const { return worker_->WorkerID(); }

  const OnlineVoiceActivityDetector *VadDetector() const {
    return vad_detector_;
  }

 private:
  const OfflineASREngineConfig &config_;
  // 记录已经接受的音频数据长度 in samples
  int32_t start_ = 0;
  int32_t session_id_;
  Scheduler *scheduler_;
  Worker *worker_;
  OnlineVoiceActivityDetector *vad_detector_;
  OfflineSessionImpl *owner_;
  std::atomic<bool> is_decode_finished_{false};
  std::atomic<bool> is_input_finished_{false};

  mutable std::mutex mutex_;
  std::vector<OfflineRecognitionResult> results_;
};

OfflineSessionImpl::OfflineSessionImpl(
    const OfflineASREngineConfig &config, int32_t session_id,
    Scheduler *scheduler, Worker *worker,
    OnlineVoiceActivityDetector *vad_detector)
    : impl_(std::make_unique<Impl>(session_id, scheduler, worker, this,
                                   vad_detector, config)) {}

OfflineSessionImpl::~OfflineSessionImpl() = default;

void OfflineSessionImpl::AcceptWaveform(int32_t sample_rate, const float *wave,
                                        int32_t num_samples,
                                        ErrorCode &error_code) {
  impl_->AcceptWaveform(sample_rate, wave, num_samples, error_code);
}

void OfflineSessionImpl::Close() { impl_->Close(); }

void OfflineSessionImpl::InputFinished() { impl_->InputFinished(); }

bool OfflineSessionImpl::IsInputFinished() const {
  return impl_->IsInputFinished();
}

// 获取当前识别结果（聚合所有已完成的片段）
std::vector<OfflineRecognitionResult> OfflineSessionImpl::GetResults() {
  return impl_->GetResults();
}

void OfflineSessionImpl::AddResult(const OfflineRecognitionResult &result) {
  impl_->AddResult(result);
}

void OfflineSessionImpl::AddResult(OfflineRecognitionResult &&result) {
  impl_->AddResult(std::move(result));
}

// 检查会话状态
bool OfflineSessionImpl::IsDecodeFinished() const {
  return impl_->IsDecodeFinished();
}

void OfflineSessionImpl::DecodeFinished() { impl_->DecodeFinished(); }

int32_t OfflineSessionImpl::SessionID() const { return impl_->SessionID(); }

int32_t OfflineSessionImpl::WorkerID() const { return impl_->WorkerID(); }

const OnlineVoiceActivityDetector *OfflineSessionImpl::VadDetector() const {
  return impl_->VadDetector();
}

}  // namespace sherpa_onnx
