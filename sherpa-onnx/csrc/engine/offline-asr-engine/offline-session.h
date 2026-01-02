#ifndef OFFLINE_ASR_ENGINE_SESSION_H_
#define OFFLINE_ASR_ENGINE_SESSION_H_

namespace sherpa_onnx {

class OfflineSession {
 public:
  virtual ~OfflineSession() = default;

  virtual void AcceptWaveform(int32_t sample_rate, const float *wave,
                              int32_t num_samples) = 0;

  virtual void Close() = 0;

 protected:
  OfflineSession() = default;
};

}  // namespace sherpa_onnx

#endif  // OFFLINE_ASR_ENGINE_SESSION_H_