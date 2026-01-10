#include "sherpa-onnx/csrc/engine/offline-asr-engine/scheduler.h"

#include <map>
#include <mutex>
#include <unordered_set>
#include <vector>

#include "sherpa-onnx/csrc/engine/offline-asr-engine/blockingconcurrentqueue.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/cpu-worker.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-asr-engine-config.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-session-impl.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/online-voice-activity-detector.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/worker.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/offline-stream.h"

namespace sherpa_onnx {

class Scheduler::Impl {
 public:
  Impl(Scheduler *owner, const OfflineASREngineConfig &config)
      : owner_(owner),
        config_(config),
        initialized_(false),
        next_session_id_(0) {}

  void Init(ErrorCode &code) {
    if (config_.enable_gpu) {
      SHERPA_ONNX_LOGE("GPU is not supported yet");
      code.error_code = ErrorCode::kInvalidArgument;
      code.error_msg = "GPU is not supported yet";
      initialized_ = false;
      return;
    } else {
      InitCPU();
      initialized_ = true;
    }
  }

  void Start(ErrorCode &code) {
    if (!initialized_) {
      code.error_code = ErrorCode::kEngineUninitialized;
      code.error_msg = "Engine is not initialized or had stopped";
      return;
    }

    code.error_code = ErrorCode::kSuccess;
  }

  OfflineSession *CreateSession(ErrorCode &code) {
    if (!initialized_) {
      code.error_code = ErrorCode::kEngineUninitialized;
      return nullptr;
    }

    std::lock_guard<std::mutex> locker(session_mutex_);

    if (idle_session_ids_.empty()) {
      code.error_code = ErrorCode::kResourceExhausted;
      return nullptr;
    }

    WorkerID worker_id =
        std::min_element(
            num_working_sessions_per_worker_.begin(),
            num_working_sessions_per_worker_.end(),
            [](const auto &l, const auto &r) { return l.second < r.second; })
            ->first;
    SessionID session_id = *idle_session_ids_.begin();
    idle_session_ids_.erase(session_id);
    sessions_[session_id] = std::make_unique<OfflineSessionImpl>(
        config_, session_id, owner_, workers_[worker_id].get(),
        vad_detectors_[session_id].get());
    num_working_sessions_per_worker_[worker_id]++;

    code.error_code = ErrorCode::kSuccess;
    return sessions_[session_id].get();
  }

  void CloseSession(int32_t session_id) {
    std::lock_guard<std::mutex> locker(session_mutex_);
    WorkerID worker_id = sessions_[session_id]->WorkerID();
    num_working_sessions_per_worker_[worker_id]--;
    idle_session_ids_.insert(session_id);
    sessions_.erase(session_id);
  }

 private:
  void InitCPU() {
    // Init Vad Detector.
    for (; next_session_id_ <
           config_.num_worker_threads * config_.max_sessions_per_worker;
         next_session_id_++) {
      vad_detectors_[next_session_id_] =
          std::make_unique<OnlineVoiceActivityDetector>(config_.vad_config);
      idle_session_ids_.emplace(next_session_id_);
    }

    // Init Recognizer.
    // for cpu decode, all worker share the same recognizer
    recognizers_[0] =
        std::make_unique<OfflineRecognizer>(config_.recognizer_config);

    // Init Task Queue.
    // every worker has its own task queue
    for (int32_t worker_id = 0; worker_id < config_.num_worker_threads;
         ++worker_id) {
      task_queues_[worker_id] = std::make_unique<TaskQueue>();
    }

    // Init Workers.
    for (int32_t worker_id = 0; worker_id < config_.num_worker_threads;
         ++worker_id) {
      workers_[worker_id] = std::make_unique<CPUWorker>(
          worker_id, config_, recognizers_[0].get(), task_queues_);
      num_working_sessions_per_worker_[worker_id] = 0;
    }
  }

 private:
  template <typename T>
  using Ptr = std::unique_ptr<T>;

  using TaskQueue = moodycamel::BlockingConcurrentQueue<WaveTask>;
  using SessionID = int32_t;
  using WorkerID = int32_t;

  Scheduler *owner_;
  OfflineASREngineConfig config_;
  bool initialized_ /* = false */;

  std::map<SessionID, Ptr<OnlineVoiceActivityDetector>> vad_detectors_;

  std::unordered_map<WorkerID, Ptr<OfflineRecognizer>> recognizers_;
  std::unordered_map<WorkerID, Ptr<Worker>> workers_;
  std::unordered_map<WorkerID, int32_t> num_working_sessions_per_worker_;
  std::unordered_map<WorkerID, Ptr<TaskQueue>> task_queues_;

  std::map<SessionID, Ptr<OfflineSessionImpl>> sessions_;
  std::unordered_set<SessionID> idle_session_ids_;
  SessionID next_session_id_;

  mutable std::mutex session_mutex_;
};

Scheduler::Scheduler(const OfflineASREngineConfig &config)
    : impl_(std::make_unique<Impl>(this, config)) {}

Scheduler::~Scheduler() = default;

void Scheduler::Init(ErrorCode &code) { impl_->Init(code); }

void Scheduler::Start(ErrorCode &code) { impl_->Start(code); }

const OfflineSession *Scheduler::CreateSession(ErrorCode &code) {
  return impl_->CreateSession(code);
}

void Scheduler::CloseSession(int32_t session_id) {
  impl_->CloseSession(session_id);
}

}  // namespace sherpa_onnx
