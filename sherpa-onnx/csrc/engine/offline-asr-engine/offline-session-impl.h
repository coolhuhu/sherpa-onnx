#ifndef OFFLINE_ASR_ENGINE_SESSION_IMPL_H_
#define OFFLINE_ASR_ENGINE_SESSION_IMPL_H_

#include "engine/offline-asr-engine/offline-session.h"

namespace sherpa_onnx {

class OfflineStream;
class OfflineRecognizer;

class OfflineSessionImpl : public OfflineSession {
 public:
  OfflineSessionImpl(OfflineRecognizer *recognizer, OfflineStream *stream)
      : recognizer_(recognizer), stream_(stream) {}

  ~OfflineSessionImpl() = default;

  void AcceptWaveform(float sample_rate, const float *wave,
                      int32_t num_samples) override;

  void Close() override;

 private:
  OfflineRecognizer *recognizer_;
  OfflineStream *stream_;
};

}  // namespace sherpa_onnx

#endif  // OFFLINE_ASR_ENGINE_SESSION_IMPL_H_