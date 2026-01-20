#ifndef OFFLINE_ASR_ENGINE_CPU_WORKER_H_
#define OFFLINE_ASR_ENGINE_CPU_WORKER_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/engine/offline-asr-engine/blockingconcurrentqueue.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-asr-engine-config.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/worker.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"

namespace sherpa_onnx {

class CPUWorker : public Worker {
 public:
  CPUWorker(
      int32_t worker_id, const OfflineASREngineConfig &config,
      OfflineRecognizer *recognizer,
      std::unique_ptr<moodycamel::BlockingConcurrentQueue<WaveTask>>
          &task_queue,
      std::unordered_map<
          int32_t, std::unique_ptr<moodycamel::BlockingConcurrentQueue<
                       std::unique_ptr<OfflineStream>>>> &shared_stream_queues);

  ~CPUWorker() override;

  void CommitWaveTask(WaveTask &&task) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // OFFLINE_ASR_ENGINE_CPU_WORKER_H_