#include "sherpa-onnx/csrc/engine/offline-asr-engine/scheduler.h"

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
      : owner_(owner), config_(config), initialized_(false) {}

  void Init(ErrorCode &code) {
    if (config_.enable_gpu) {
      SHERPA_ONNX_LOGE("GPU is not supported yet");
      code = kInvalidArgument;
      initialized_ = false;
      return;
    } else {
      InitCPU();
    }
  }

  void Start(ErrorCode &code) {
    if (!initialized_) {
      code = kEngineUninitialized;
      return;
    }

    code = kSuccess;
  }

  OfflineSession *CreateSession(ErrorCode &code) {
    if (!initialized_) {
      code = kEngineUninitialized;
      return nullptr;
    }

    std::lock_guard<std::mutex> locker(session_mutex_);

    if (idle_sessions_id_.empty()) {
      code = kResourceExhausted;
      return nullptr;
    }

    int32_t worker_id =
        std::min_element(num_working_sessions_per_worker_.begin(),
                         num_working_sessions_per_worker_.end()) -
        num_working_sessions_per_worker_.begin();

    SessionID session_id = *idle_sessions_id_.begin();
    idle_sessions_id_.erase(session_id);
    sessions_[session_id] = std::make_unique<OfflineSessionImpl>(
        session_id, owner_, workers_[worker_id].get(), config_);

    num_working_sessions_per_worker_[worker_id]++;

    code = kSuccess;
    return sessions_[session_id].get();
  }

  void CloseSession(int32_t session_id) {
    std::lock_guard<std::mutex> locker(session_mutex_);
    WorkerID worker_id = std::get<0>(session_manager_[session_id])->WorkerId();
    num_working_sessions_per_worker_[worker_id]--;
    idle_sessions_id_.insert(session_id);
    sessions_[session_id].reset(nullptr);
  }

 private:
  void InitCPU() {
    // Init Vad Detector.

    // Init Recognizer.
    // for cpu decode, all worker share the same recognizer
    recognizers_.emplace_back(
        std::make_unique<OfflineRecognizer>(config_.recognizer_config));

    // Init Task Queue.
    // every worker has its own task queue
    vad_task_queues_.reserve(config_.num_workers);
    vad_task_queues_ptr_.resize(config_.num_workers, nullptr);
    // TODO(lianghu): queue size should be configurable
    for (int i = 0; i < config_.num_workers; ++i) {
      vad_task_queues_.emplace_back(std::make_unique<TaskQueue<VadTask>>(
          (std::ceil(30 /
                     moodycamel::BlockingConcurrentQueue<VadTask>::BLOCK_SIZE) +
           1) *
          30 * moodycamel::BlockingConcurrentQueue<VadTask>::BLOCK_SIZE));
    }
    for (int i = 0; i < config_.num_workers; ++i) {
      vad_task_queues_ptr_[i] = vad_task_queues_[i].get();
    }

    decode_task_queues_.reserve(config_.num_workers);
    decode_task_queues_ptr_.resize(config_.num_workers, nullptr);
    for (int i = 0; i < config_.num_workers; ++i) {
      decode_task_queues_.emplace_back(std::make_unique<TaskQueue<DecodeTask>>(
          (std::ceil(
               30 /
               moodycamel::BlockingConcurrentQueue<DecodeTask>::BLOCK_SIZE) +
           1) *
          30 * moodycamel::BlockingConcurrentQueue<DecodeTask>::BLOCK_SIZE));
    }
    for (int i = 0; i < config_.num_workers; ++i) {
      decode_task_queues_ptr_[i] = decode_task_queues_[i].get();
    }

    // Init Workers.
    workers_.reserve(config_.num_workers);
    for (int32_t i = 0; i < config_.num_workers; ++i) {
      workers_.emplace_back(std::make_unique<CPUWorker>(
          i, config_, recognizers_[0].get(), vad_task_queues_ptr_,
          decode_task_queues_ptr_));
    }
    num_working_sessions_per_worker_.resize(config_.num_workers, 0);

    // Init Sessions.
  }

 private:
  template <typename T>
  using Ptr = std::unique_ptr<T>;

  template <typename T>
  using TaskQueue = moodycamel::BlockingConcurrentQueue<T>;

  using SessionID = int32_t;
  using WorkerID = int32_t;

  Scheduler *owner_;
  OfflineASREngineConfig config_;
  bool initialized_ /* = false */;

  std::vector<Ptr<OnlineVoiceActivityDetector>> vad_detectors_;

  std::vector<Ptr<OfflineRecognizer>> recognizers_;
  std::vector<Ptr<Worker>> workers_;
  std::vector<int32_t> num_working_sessions_per_worker_;

  std::vector<Ptr<TaskQueue<VadTask>>> vad_task_queues_;
  std::vector<TaskQueue<VadTask> *> vad_task_queues_ptr_;
  std::vector<Ptr<TaskQueue<DecodeTask>>> decode_task_queues_;
  std::vector<TaskQueue<DecodeTask> *> decode_task_queues_ptr_;

  std::vector<Ptr<OfflineSession>> sessions_;
  // [0, sessions_.size() - 1]
  std::unordered_set<SessionID> idle_sessions_id_;

  std::unordered_map<SessionID,
                     std::tuple<Worker *, OnlineVoiceActivityDetector *>>
      session_manager_;

  std::mutex session_mutex_;
};

Scheduler::Scheduler(const OfflineASREngineConfig &config)
    : impl_(std::make_unique<Impl>(this, config)) {}

Scheduler::~Scheduler() = default;

void Scheduler::Init(ErrorCode &code) { impl_->Init(code); }

void Scheduler::Start(ErrorCode &code) { impl_->Start(code); }

OfflineSession *Scheduler::CreateSession(ErrorCode &code) {
  return impl_->CreateSession(code);
}

void Scheduler::CloseSession(int32_t session_id) {
  impl_->CloseSession(session_id);
}

}  // namespace sherpa_onnx
