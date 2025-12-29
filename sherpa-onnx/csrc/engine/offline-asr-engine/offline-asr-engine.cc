#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-asr-engine.h"

#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-asr-engine-impl.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-session.h"

namespace sherpa_onnx {

OfflineASREngine::OfflineASREngine(const OfflineASREngineConfig &config)
    : impl_(std::make_unique<OfflineASREngineImpl>(config)) {}

void OfflineASREngine::Start(ErrorCode &code) {}

void OfflineASREngine::Shutdown() {}

OfflineSession *OfflineASREngine::CreateSession(ErrorCode &code) {
  return impl_->CreateSession(code);
}

}  // namespace sherpa_onnx
