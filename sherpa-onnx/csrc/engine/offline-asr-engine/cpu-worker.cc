#include "sherpa-onnx/csrc/engine/offline-asr-engine/cpu-worker.h"

#include <atomic>
#include <thread>

#include "sherpa-onnx/csrc/engine/offline-asr-engine/blockingconcurrentqueue.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-asr-engine-config.h"

namespace sherpa_onnx {

class CPUWorker::Impl {
 public:
  Impl(CPUWorker *owner, const OfflineASREngineConfig &config,
       OfflineRecognizer *recognizer,
       std::unordered_map<
           int32_t,
           std::unique_ptr<moodycamel::BlockingConcurrentQueue<WaveTask>>>
           &workers_task_queues)
      : owner_(owner),
        config_(config),
        recognizer_(recognizer),
        workers_task_queues_(workers_task_queues),
        stop_(true) {}

  ~Impl() { Stop(); }

  void AddSession() { num_sessions_++; }

  void RemoveSession() { num_sessions_--; }

  void CommitWaveTask(WaveTask &&task) {}

  void Start() {
    task_queue_ = workers_task_queues_[owner_->WorkerID()].get();
    stop_ = false;
    thread_ = std::thread(&CPUWorker::Impl::Pipeline, this);
  }

  void Stop() {
    if (!stop_) {
      stop_ = true;

      if (thread_.joinable()) {
        thread_.join();
      }
    }
  }

 private:
  void Pipeline() {
    while (!stop_) {
      if (config_.enable_vad) {
        PipelineWithVAD();
      } else {
        PipelineWithoutVAD();
      }
    }

    // TODO(lianghu): how to corrently empty queue?
  }

  void PipelineWithoutVAD() {
    int32_t wait_time = 10000;  // in microseconds

    std::vector<WaveTask> tasks(config_.max_sessions_per_worker);
    int32_t num_task = task_queue_->wait_dequeue_bulk_timed(
        tasks.begin(), config_.max_sessions_per_worker, wait_time);
    if (num_task <= 0) {
      return;
    }
  }

  void PipelineWithVAD() {}

 private:
  CPUWorker *owner_;
  const OfflineASREngineConfig &config_;
  OfflineRecognizer *recognizer_;
  std::unordered_map<
      int32_t, std::unique_ptr<moodycamel::BlockingConcurrentQueue<WaveTask>>>
      &workers_task_queues_;
  moodycamel::BlockingConcurrentQueue<WaveTask> *task_queue_;

  std::atomic<int32_t> num_sessions_;

  std::atomic<bool> stop_ /* = false */;
  std::thread thread_;
};

CPUWorker::CPUWorker(
    int32_t worker_id, const OfflineASREngineConfig &config,
    OfflineRecognizer *recognizer,
    std::unordered_map<
        int32_t, std::unique_ptr<moodycamel::BlockingConcurrentQueue<WaveTask>>>
        &workers_task_queues)
    : Worker(worker_id),
      impl_(std::make_unique<Impl>(this, config, recognizer,
                                   workers_task_queues)) {}

CPUWorker::~CPUWorker() = default;

void CPUWorker::AddSession() { impl_->AddSession(); }

void CPUWorker::RemoveSession() { impl_->RemoveSession(); }

void CPUWorker::CommitWaveTask(WaveTask &&task) {
  impl_->CommitWaveTask(std::move(task));
}

}  // namespace sherpa_onnx
