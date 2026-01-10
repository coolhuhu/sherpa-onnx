#ifndef OFFLINE_ASR_ENGINE_CONFIG_H_
#define OFFLINE_ASR_ENGINE_CONFIG_H_

#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/vad-model-config.h"

namespace sherpa_onnx {

struct OfflineASREngineConfig {
  int32_t sample_rate = 16000;
  int32_t num_worker_threads = 4;  // 工作线程数（ASR推理）
  int32_t max_sessions_per_worker = 4;

  // VAD配置
  // 当enable_vad=true时：
  //   - IO线程会实时处理音频，进行VAD检测
  //   - 检测到的语音段会被切分成多个片段并行处理
  //   - 适合长音频、有静音段的场景
  // 当enable_vad=false时：
  //   - 不启动IO线程，不进行VAD检测
  //   - 整个音频作为一个片段处理
  //   - 适合短音频或已经预先分割好的音频
  bool enable_vad = true;  // 是否启用VAD

  int32_t max_accept_waveform_size =
      sample_rate * 300;  // AcceptWaveform单次最大采样点数
  int32_t max_model_input_samples =
      sample_rate *
      30;  // 模型允许推理的最大采样点数，例如 whisper 最大支持30s的音频

  // 批处理配置
  int32_t max_batch_size = 8;  // 最大批量大小

  // 调度配置
  bool enable_task_stealing = true;  // 启用任务窃取

  // GPU配置（预留，未来版本实现）
  bool enable_gpu = false;                    // 是否启用GPU（预留）
  std::vector<int32_t> gpu_device_ids = {0};  // GPU设备ID列表（预留）

  // 模型配置
  VadModelConfig vad_config;                  // VAD模型配置
  OfflineRecognizerConfig recognizer_config;  // ASR模型配置

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // OFFLINE_ASR_ENGINE_CONFIG_H_