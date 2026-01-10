#ifndef OFFLINE_ASR_ENGINE_SESSION_H_
#define OFFLINE_ASR_ENGINE_SESSION_H_

#include <functional>
#include <memory>

#include "sherpa-onnx/csrc/engine/offline-asr-engine/error-code.h"
#include "sherpa-onnx/csrc/offline-stream.h"

namespace sherpa_onnx {

class OfflineSession {
 public:
  ~OfflineSession() = default;

  virtual void AcceptWaveform(int32_t sample_rate, const float *wave,
                              int32_t num_samples, ErrorCode &error_code) = 0;

  // 标记输入结束（不再有音频数据）
  // 当enable_vad=false时，只有调用此方法后才会开始处理
  virtual void InputFinished() = 0;

  // 获取当前识别结果（聚合所有已完成的片段）
  virtual std::vector<OfflineRecognitionResult> GetResults() = 0;

  // 检查会话状态
  virtual bool IsDecodeFinished() const = 0;  // 是否完成所有处理

  // 显式关闭会话
  virtual void Close() = 0;

  virtual int32_t SessionID() const = 0;

 protected:
  OfflineSession() = default;
};

}  // namespace sherpa_onnx

#endif  // OFFLINE_ASR_ENGINE_SESSION_H_
