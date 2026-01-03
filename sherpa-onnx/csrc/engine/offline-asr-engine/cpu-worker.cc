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
       const std::vector<moodycamel::BlockingConcurrentQueue<VadTask> *>
           &vad_task_queues,
       const std::vector<moodycamel::BlockingConcurrentQueue<DecodeTask> *>
           &decode_task_queues)
      : owner_(owner),
        config_(config),
        recognizer_(recognizer),
        other_vad_task_queues_(vad_task_queues),
        other_decode_task_queues_(decode_task_queues) {}

  ~Impl() { Stop(); }

  void AddSession() { num_sessions_++; }

  void RemoveSession() { num_sessions_--; }

  void CommitVadTask(VadTask &&task) {
    vad_task_queue_->enqueue(std::move(task));
  }

  void CommitDecodeTask(DecodeTask &&task) {
    decode_task_queue_->enqueue(std::move(task));
  }

  void Start() {
    vad_task_queue_ = other_vad_task_queues_[owner_->worker_id_];
    decode_task_queue_ = other_decode_task_queues_[owner_->worker_id_];

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
    bool use_vad = false;
    while (!stop_) {
      if (config_.use_vad) {
        VadTask vad_task;
        if (vad_task_queue_->try_dequeue(vad_task)) {
        }
      }

      int64_t wait_time = 1000000;  // microseconds
      DecodeTask task;
      if (decode_task_queue_->wait_dequeue_timed(task, wait_time)) {
        std::unique_ptr<OfflineStream> stream = recognizer_->CreateStream();
        stream->AcceptWaveform(task.sample_rate, task.samples.data(),
                               task.samples.size());

        recognizer_->DecodeStream(stream.get());
        auto result = stream->GetResult();
      }

      if (config_.enable_task_stealing) {
        if (num_sessions_ == 0) {
          // TODO(lianghu): task stealing
        }
      }
    }

    // TODO(lianghu): how to corrently empty queue?
  }

 private:
  CPUWorker *owner_;
  const OfflineASREngineConfig &config_;
  OfflineRecognizer *recognizer_;

  std::vector<moodycamel::BlockingConcurrentQueue<VadTask> *>
      other_vad_task_queues_;
  std::vector<moodycamel::BlockingConcurrentQueue<DecodeTask> *>
      other_decode_task_queues_;
  moodycamel::BlockingConcurrentQueue<VadTask> *vad_task_queue_;
  moodycamel::BlockingConcurrentQueue<DecodeTask> *decode_task_queue_;
  std::atomic<int32_t> num_sessions_;

  std::atomic<bool> stop_ /* = false */;
  std::thread thread_;
};

CPUWorker::CPUWorker(
    int32_t worker_id, const OfflineASREngineConfig &config,
    OfflineRecognizer *recognizer,
    const std::vector<moodycamel::BlockingConcurrentQueue<VadTask> *>
        &vad_task_queues,
    const std::vector<moodycamel::BlockingConcurrentQueue<DecodeTask> *>
        &decode_task_queues)
    : Worker(worker_id),
      impl_(std::make_unique<Impl>(this, config, recognizer, vad_task_queues,
                                   decode_task_queues)) {}

CPUWorker::~CPUWorker() = default;

void CPUWorker::AddSession() { impl_->AddSession(); }

void CPUWorker::RemoveSession() { impl_->RemoveSession(); }

void CPUWorker::CommitVadTask(VadTask &&task) {
  impl_->CommitVadTask(std::move(task));
}

void CPUWorker::CommitDecodeTask(DecodeTask &&task) {
  impl_->CommitDecodeTask(std::move(task));
}

}  // namespace sherpa_onnx
