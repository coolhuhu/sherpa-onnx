#include "sherpa-onnx/csrc/engine/offline-asr-engine/cpu-worker.h"

#include <algorithm>
#include <atomic>
#include <random>
#include <thread>
#include <unordered_set>

#include "sherpa-onnx/csrc/engine/offline-asr-engine/blockingconcurrentqueue.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-asr-engine-config.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/online-voice-activity-detector.h"
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
           &shared_stream_queues,
       std::unordered_map<int32_t, std::atomic<int32_t>>
           &num_working_sessions_per_worker)
      : owner_(owner),
        config_(config),
        recognizer_(recognizer),
        task_queue_(task_queue),
        shared_segment_queues_(shared_stream_queues),
        stop_(true),
        num_working_sessions_per_worker_(num_working_sessions_per_worker) {}

  ~Impl() { Stop(); }

  void CommitWaveTask(WaveTask &&task) {
    task_queue_->enqueue(std::move(task));
  }

  void Start() {
    segment_queue_ = shared_segment_queues_[owner_->WorkerID()].get();
    for (auto &worker : num_working_sessions_per_worker_) {
      if (worker.first != owner_->WorkerID()) {
        steal_worker_ids_.push_back(worker.first);
      }
    }

    stop_ = false;
    frontend_thread_ = std::thread(&CPUWorker::Impl::Pipeline, this);
    decode_thread_ = std::thread(&CPUWorker::Impl::Decode, this);
  }

  void Stop() {
    if (!stop_) {
      stop_ = true;

      if (frontend_thread_.joinable()) {
        frontend_thread_.join();
      }

      if (decode_thread_.joinable()) {
        decode_thread_.join();
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
      return;
    }

    // build offline stream and push to segment queue
    std::vector<std::unique_ptr<OfflineStream>> streams;
    for (int32_t i = 0; i < num_task; ++i) {
      std::unique_ptr<OfflineStream> stream = recognizer_->CreateStream();
      stream->AcceptWaveform(tasks[i].sample_rate, tasks[i].samples.data(),
                             tasks[i].samples.size());

      tasks[i].session->IncrementSegmentId();

      SegmentTask segment_task;
      // Without VAD, segment_id is always 1
      segment_task.segment_id = tasks[i].session->SegmentID();
      segment_task.start_time =
          static_cast<float>(tasks[i].start) / tasks[i].sample_rate;
      segment_task.end_time =
          static_cast<float>(tasks[i].start + tasks[i].samples.size()) /
          tasks[i].sample_rate;
      segment_task.stream = std::move(stream);
      segment_task.session = tasks[i].session;

      if (tasks[i].session->IsInputFinished(tasks[i].task_id)) {
        tasks[i].session->SetLastSegmentId(segment_task.segment_id);
      }

      segment_queue_->enqueue(std::move(segment_task));
    }
  }

  void PipelineWithVAD() {
    int64_t wait_time = 10000;  // in microseconds

    std::vector<WaveTask> tasks(config_.max_sessions_per_worker);
    int32_t num_task = task_queue_->wait_dequeue_bulk_timed(
        tasks.begin(), config_.max_sessions_per_worker, wait_time);

    /// 没有新的数据到来
    if (num_task <= 0) {
      return;
    }

    /// 1. VAD
    /// 2. stream->AcceptWaveform
    for (int i = 0; i < num_task; ++i) {
      WaveTask &task = tasks[i];

      OfflineSessionImpl *session = task.session;

      if (session->IsInputFinished(task.task_id)) {
        /// 在开启 VAD 的情况下，当 session 不再调用 AcceptWaveform 接口，
        /// 然后调用 InputFinished 接口，告知引擎，当前 session
        /// 已经完成所有输入。 InputFinished 接口内部会向队列中提交一个空的
        /// Task， 也即 session 通过队列提交给引擎的最后一个 Task， 这个空的
        /// Task 中是不包含有效的语音数据的。 因此当调用 InputFinished
        /// 接口检测到这是 session 提交给队列的最后一个 Task 时，设置 session 的
        /// last_segment_id 为当前 segment_id， 表示 session 的所有 segment
        /// 均已提交给引擎处理， 对于 session
        /// 的这个最后的空的Task，不进行任何处理。
        session->SetLastSegmentId(session->SegmentID());
        continue;
      }

      OnlineVoiceActivityDetector *vad = session->VadDetector();
      vad->AcceptWaveform(task.samples);
      if (!vad->Empty()) {
        std::vector<VadSpeechSegment> speech_segments = vad->GetSpeechSegment();
        for (auto &s : speech_segments) {
          std::unique_ptr<OfflineStream> stream = recognizer_->CreateStream();
          stream->AcceptWaveform(task.sample_rate, s.samples.data(),
                                 s.samples.size());

          session->IncrementSegmentId();

          SegmentTask segment_task;
          segment_task.segment_id = session->SegmentID();
          segment_task.start_time =
              static_cast<float>(s.start) / task.sample_rate;
          segment_task.end_time =
              static_cast<float>(s.start + s.samples.size()) / task.sample_rate;
          segment_task.stream = std::move(stream);

          segment_queue_->enqueue(std::move(segment_task));
        }
      }
    }
  }

  void Decode() {
    while (!stop_) {
      int64_t wait_time = 10000;  // in microseconds

      SegmentTask task;
      bool task_ready = segment_queue_->wait_dequeue_timed(task, wait_time);
      if (!task_ready) {
        TaskSteal();
        continue;
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
  }

  void TaskSteal() {
    if (!(config_.enable_task_stealing && IsIdle())) {
      return;
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(steal_worker_ids_.begin(), steal_worker_ids_.end(), g);
    for (int32_t worker_id : steal_worker_ids_) {
      if (num_working_sessions_per_worker_[worker_id] == 0) {
        continue;
      }

      SegmentQueue *steal_queue = shared_segment_queues_[worker_id].get();

      SegmentTask task;
      while (steal_queue->try_dequeue(task)) {
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

        if (!IsIdle()) {
          return;
        }
      }

      if (!IsIdle()) {
        return;
      }
    }
  }

  bool IsIdle() const {
    return num_working_sessions_per_worker_[owner_->WorkerID()] == 0;
  }

 private:
  using TaskQueue = moodycamel::BlockingConcurrentQueue<WaveTask>;
  using SegmentQueue = moodycamel::BlockingConcurrentQueue<SegmentTask>;

  CPUWorker *owner_;
  const OfflineASREngineConfig &config_;
  OfflineRecognizer *recognizer_;
  std::unique_ptr<TaskQueue> &task_queue_;
  std::unordered_map<int32_t, std::unique_ptr<SegmentQueue>>
      &shared_segment_queues_;
  SegmentQueue *segment_queue_;

  std::unordered_map<int32_t, std::atomic<int32_t>>
      &num_working_sessions_per_worker_;
  std::vector<int32_t> steal_worker_ids_;

  std::atomic<bool> stop_ /* = false */;
  std::thread frontend_thread_;
  std::thread decode_thread_;
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

void CPUWorker::CommitWaveTask(WaveTask &&task) {
  impl_->CommitWaveTask(std::move(task));
}

}  // namespace sherpa_onnx
