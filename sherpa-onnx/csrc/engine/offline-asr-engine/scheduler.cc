#include "sherpa-onnx/csrc/engine/offline-asr-engine/scheduler.h"

#include <mutex>
#include <vector>

#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/offline-stream.h"

namespace sherpa_onnx {

class Scheduler::Impl {
 public:
  Impl() = default;

  OfflineSession *CreateSession(ErrorCode &code) { return nullptr; }

 private:
  std::vector<OfflineRecognizer> recognizers_;
  std::vector<std::unique_ptr<OfflineStream>> streams_;

  std::mutex session_mutex_;
};

OfflineSession *Scheduler::CreateSession(ErrorCode &code) {
  return impl_->CreateSession(code);
}

}  // namespace sherpa_onnx
