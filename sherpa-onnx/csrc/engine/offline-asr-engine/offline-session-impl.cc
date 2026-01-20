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
        last_task_id_(-1),
        task_id_(0),
        segment_id_(0),
        num_finfished_segments_(0),
        last_segment_id_(-1),
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

    if (!config_.use_vad &&
        num_samples + start_ > config_.max_model_input_samples) {
      error_code.error_code = ErrorCode::kInvalidArgument;
      error_code.error_msg =
          "Invalid argument in AcceptWaveform: input wave is too long";
      return;
    }

    if (config_.use_vad) {
      task_id_++;
      WaveTask task(task_id_, sample_rate, start_, wave, num_samples, owner_);
      start_ += num_samples;
      worker_->CommitWaveTask(std::move(task));
    } else {
      buffer_.insert(buffer_.end(), wave, wave + num_samples);
      start_ += num_samples;
    }
  }

  void Close() { scheduler_->CloseSession(session_id_); }

  void InputFinished() {
    // TODO: 限制该函数只能被调用一次

    task_id_++;
    last_task_id_ = task_id_;

    if (config_.use_vad) {
      WaveTask task;
      task.task_id = task_id_;
      task.start = -1;
      task.sample_rate = config_.sample_rate;
      task.session = owner_;

      worker_->CommitWaveTask(std::move(task));
    } else {
      WaveTask task;
      task.task_id = task_id_;
      task.start = start_;
      task.sample_rate = config_.sample_rate;
      task.samples = std::move(buffer_);
      task.session = owner_;

      worker_->CommitWaveTask(std::move(task));
    }
  }

  bool IsInputFinished(int32_t task_id) const { last_task_id_ == task_id; }

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

  bool IsDecodeFinished() const {
    return num_finfished_segments_ == last_segment_id_;
  }

  void IncrementFinishedSegment() { ++num_finfished_segments_; }

  void IncrementSegmentId() { ++segment_id_; }

  int32_t SegmentID() const { return segment_id_; }

  void SetLastSegmentId(int32_t id) { last_segment_id_ = id; }

  int32_t SessionID() const { return session_id_; }

  int32_t WorkerID() const { return worker_->WorkerID(); }

  OnlineVoiceActivityDetector *VadDetector() const { return vad_detector_; }

 private:
  const OfflineASREngineConfig &config_;
  // 记录已经接受的音频数据长度 in samples
  int32_t start_ = 0;
  int32_t session_id_;
  Scheduler *scheduler_;
  Worker *worker_;
  OnlineVoiceActivityDetector *vad_detector_;
  OfflineSessionImpl *owner_;
  int32_t last_task_id_;
  int32_t task_id_;
  std::atomic<int32_t> segment_id_;
  std::atomic<int32_t> num_finfished_segments_;
  std::atomic<int32_t> last_segment_id_;
  std::atomic<bool> is_input_finished_{false};

  std::vector<float> buffer_;

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

bool OfflineSessionImpl::IsInputFinished(int32_t task_id) const {
  return impl_->IsInputFinished(task_id);
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

int32_t OfflineSessionImpl::SessionID() const { return impl_->SessionID(); }

int32_t OfflineSessionImpl::WorkerID() const { return impl_->WorkerID(); }

OnlineVoiceActivityDetector *OfflineSessionImpl::VadDetector() const {
  return impl_->VadDetector();
}

void OfflineSessionImpl::IncrementFinishedSegment() {
  impl_->IncrementFinishedSegment();
}

void OfflineSessionImpl::IncrementSegmentId() { impl_->IncrementSegmentId(); }

int32_t OfflineSessionImpl::SegmentID() const { return impl_->SegmentID(); }

void OfflineSessionImpl::SetLastSegmentId(int32_t id) {
  impl_->SetLastSegmentId(id);
}

}  // namespace sherpa_onnx
