#include "sherpa-onnx/csrc/engine/offline-asr-engine/cpu-worker.h"

#include <atomic>
#include <thread>
#include <unordered_set>

#include "sherpa-onnx/csrc/engine/offline-asr-engine/blockingconcurrentqueue.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-asr-engine-config.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"

namespace sherpa_onnx {

class CPUWorker::Impl {
 public:
  Impl(CPUWorker *owner, const OfflineASREngineConfig &config,
       OfflineRecognizer *recognizer,
       std::unique_ptr<moodycamel::BlockingConcurrentQueue<WaveTask>>
           &task_queue,
       std::unordered_map<
           int32_t,
           std::unique_ptr<moodycamel::BlockingConcurrentQueue<SegmentTask>>>
           &shared_stream_queues)
      : owner_(owner),
        config_(config),
        recognizer_(recognizer),
        task_queue_(task_queue),
        shared_segment_queues_(shared_stream_queues),
        stop_(true) {}

  ~Impl() { Stop(); }

  void AddSession(OfflineSessionImpl *session) {
    std::unique_lock<std::mutex> lock(mutex_);
    sessions_.emplace(session);
  };

  void CommitWaveTask(WaveTask &&task) {
    task_queue_->enqueue(std::move(task));
  }

  void Start() {
    segment_queue_ = shared_segment_queues_[owner_->WorkerID()].get();
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
      if (config_.use_vad) {
        PipelineWithVAD();
      } else {
        PipelineWithoutVAD();
      }
    }

    // TODO(lianghu): how to corrently empty queue?
  }

  void PipelineWithoutVAD() {
    int64_t wait_time = 10000;  // in microseconds

    std::vector<WaveTask> tasks(config_.max_sessions_per_worker);
    int32_t num_task = task_queue_->wait_dequeue_bulk_timed(
        tasks.begin(), config_.max_sessions_per_worker, wait_time);
    if (num_task <= 0) {
      int32_t num_segments = segment_queue_->size_approx();
      for (int i = 0; i < num_segments; ++i) {
        Decode();
      }

      // task stealing
      TaskSteal();

      return;
    }

    // build offline stream and push to segment queue
    std::vector<std::unique_ptr<OfflineStream>> streams;
    for (int32_t i = 0; i < num_task; ++i) {
      std::unique_ptr<OfflineStream> stream = recognizer_->CreateStream();
      stream->AcceptWaveform(tasks[i].sample_rate, tasks[i].samples.data(),
                             tasks[i].samples.size());

      SegmentTask segment_task;
      // Without VAD, segment_id is always 0
      segment_task.segment_id = 0;
      segment_task.start_time =
          static_cast<float>(tasks[i].start) / tasks[i].sample_rate;
      segment_task.end_time =
          static_cast<float>(tasks[i].start + tasks[i].samples.size()) /
          tasks[i].sample_rate;
      segment_task.stream = std::move(stream);
      segment_task.session = tasks[i].session;

      segment_queue_->enqueue(std::move(segment_task));
    }

    // decoding
    int32_t num_segments = segment_queue_->size_approx();
    for (int i = 0; i < num_segments; ++i) {
      Decode();
    }

    // task stealing
    TaskSteal();
  }

  void PipelineWithVAD() {
    int64_t wait_time = 10000;  // in microseconds

    std::vector<WaveTask> tasks(config_.max_sessions_per_worker);
    int32_t num_task = task_queue_->wait_dequeue_bulk_timed(
        tasks.begin(), config_.max_sessions_per_worker, wait_time);
    if (num_task <= 0) {
      int32_t num_segments = segment_queue_->size_approx();
      for (int i = 0; i < num_segments; ++i) {
        Decode();
      }

      // task stealing
      TaskSteal();

      return;
    }
  }

  void Decode() {
    SegmentTask task;
    bool task_ready = segment_queue_->try_dequeue(task);
    if (!task_ready) {
      return;
    }
    OfflineStream *stream = task.stream.get();
    OfflineSessionImpl *session = task.session;
    recognizer_->DecodeStream(stream);
    OfflineRecognitionResult result = stream->GetResult();

    // update start time and end time
    result.segment_id = task.segment_id;
    result.start_time = task.start_time;
    result.end_time = task.end_time;
    session->AddResult(std::move(result));

    // update session's num_finished_segments
    session->IncrementFinishedSegment();
  }

  void TaskSteal() {
    if (config_.enable_task_stealing && sessions_.size() == 0) {
    }
  }

 private:
  CPUWorker *owner_;
  const OfflineASREngineConfig &config_;
  OfflineRecognizer *recognizer_;
  std::unique_ptr<moodycamel::BlockingConcurrentQueue<WaveTask>> &task_queue_;
  std::unordered_map<
      int32_t,
      std::unique_ptr<moodycamel::BlockingConcurrentQueue<SegmentTask>>>
      &shared_segment_queues_;
  moodycamel::BlockingConcurrentQueue<SegmentTask> *segment_queue_;
  std::unordered_set<OfflineSessionImpl *> sessions_;

  std::atomic<int32_t> num_sessions_;

  std::atomic<bool> stop_ /* = false */;
  std::thread thread_;
  std::mutex mutex_;
};

CPUWorker::CPUWorker(
    int32_t worker_id, const OfflineASREngineConfig &config,
    OfflineRecognizer *recognizer,
    std::unique_ptr<moodycamel::BlockingConcurrentQueue<WaveTask>> &task_queue,
    std::unordered_map<
        int32_t, std::unique_ptr<moodycamel::BlockingConcurrentQueue<
                     std::unique_ptr<OfflineStream>>>> &shared_stream_queues)
    : Worker(worker_id),
      impl_(std::make_unique<Impl>(this, config, recognizer, task_queue,
                                   shared_stream_queues)) {}

CPUWorker::~CPUWorker() = default;

void CPUWorker::AddSession(OfflineSessionImpl *session) {
  impl_->AddSession(session);
}

void CPUWorker::CommitWaveTask(WaveTask &&task) {
  impl_->CommitWaveTask(std::move(task));
}

}  // namespace sherpa_onnx
