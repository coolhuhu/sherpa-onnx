#ifndef OFFLINE_ASR_ENGINE_WORKER_H_
#define OFFLINE_ASR_ENGINE_WORKER_H_

#include "sherpa-onnx/csrc/engine/offline-asr-engine/task.h"

namespace sherpa_onnx {

class Worker {
 public:
  Worker(int32_t worker_id) : worker_id_(worker_id) {}

  virtual ~Worker() = default;

  virtual void AddSession() = 0;

  virtual void RemoveSession() = 0;

  virtual void CommitVadTask(VadTask &&task) = 0;

  virtual void CommitDecodeTask(DecodeTask &&task) = 0;

  int32_t WorkerId() const { return worker_id_; }

 protected:
  int32_t worker_id_;
};

}  // namespace sherpa_onnx

#endif  // OFFLINE_ASR_ENGINE_WORKER_H