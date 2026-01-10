#ifndef OFFLINE_ASR_ENGINE_SESSION_IMPL_H_
#define OFFLINE_ASR_ENGINE_SESSION_IMPL_H_

#include <memory>

#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-asr-engine-config.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-session.h"

namespace sherpa_onnx {

class Worker;
class Scheduler;
class OnlineVoiceActivityDetector;

class OfflineSessionImpl : public OfflineSession {
 public:
  OfflineSessionImpl(const OfflineASREngineConfig &config, int32_t session_id,
                     Scheduler *scheduler, Worker *worker,
                     OnlineVoiceActivityDetector *vad_detector);

  ~OfflineSessionImpl();

  /// 外部的接口
  ///
  void AcceptWaveform(int32_t sample_rate, const float *wave,
                      int32_t num_samples, ErrorCode &error_code) override;

  void InputFinished() override;

  // 获取当前识别结果（聚合所有已完成的片段）
  std::vector<OfflineRecognitionResult> GetResults() override;

  bool IsDecodeFinished() const override;  // 是否完成所有处理

  void Close() override;

  int32_t SessionID() const override;

  /// 内部接口
  ///
  bool IsInputFinished() const;

  void DecodeFinished();

  void AddResult(const OfflineRecognitionResult &result);
  void AddResult(OfflineRecognitionResult &&result);

  int32_t WorkerID() const;

  const OnlineVoiceActivityDetector *VadDetector() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // OFFLINE_ASR_ENGINE_SESSION_IMPL_H_