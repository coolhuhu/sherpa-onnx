// sherpa-onnx/csrc/engine/offline-asr-engine/engine-example.cc
//
// 这是一个使用 OfflineASREngine 的示例程序
// 演示如何配置引擎、创建会话、处理音频数据

#include <stdio.h>

#include <chrono>
#include <memory>

#include "sherpa-onnx/csrc/engine/offline-asr-engine/error-code.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-asr-engine.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-asr-engine-config.h"
#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-session.h"
#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/wave-reader.h"

int main(int argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
OfflineASREngine 使用示例

这个程序演示如何使用 OfflineASREngine 接口进行离线语音识别。

使用方法 (以 Paraformer 模型为例):

  ./engine-example \
    --tokens=/path/to/tokens.txt \
    --paraformer=/path/to/model.onnx \
    --num-threads=2 \
    --num-workers=4 \
    /path/to/audio.wav

使用方法 (以 Whisper 模型为例):

  ./engine-example \
    --whisper-encoder=/path/to/encoder.onnx \
    --whisper-decoder=/path/to/decoder.onnx \
    --tokens=/path/to/tokens.txt \
    --num-threads=2 \
    --num-workers=4 \
    /path/to/audio.wav

注意: audio.wav 应该是单通道、16位 PCM 编码的 WAV 文件。
)usage";

  sherpa_onnx::ParseOptions po(kUsageMessage);

  // 创建引擎配置
  sherpa_onnx::OfflineASREngineConfig engine_config;
  sherpa_onnx::OfflineRecognizerConfig &recognizer_config =
      engine_config.recognizer_config;

  // 注册配置选项
  recognizer_config.Register(&po);

  // 添加引擎特定的选项
  po.Register("num-workers", &engine_config.num_workers,
              "Number of worker threads for parallel processing");
  po.Register("enable-gpu", &engine_config.enable_gpu,
              "Enable GPU acceleration (if available)");
  po.Register("use-vad", &engine_config.use_vad,
              "Use Voice Activity Detection");

  po.Read(argc, argv);

  if (po.NumArgs() < 1) {
    fprintf(stderr, "Error: Please provide at least 1 wave file.\n\n");
    po.PrintUsage();
    return EXIT_FAILURE;
  }

  // 验证配置
  if (!recognizer_config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }

  fprintf(stderr, "%s\n", recognizer_config.ToString().c_str());
  fprintf(stderr, "Engine config:\n");
  fprintf(stderr, "  num_workers: %d\n", engine_config.num_workers);
  fprintf(stderr, "  enable_gpu: %s\n", engine_config.enable_gpu ? "true" : "false");
  fprintf(stderr, "  use_vad: %s\n", engine_config.use_vad ? "true" : "false");

  // 创建引擎
  fprintf(stderr, "\nCreating OfflineASREngine...\n");
  const auto begin_init = std::chrono::steady_clock::now();

  sherpa_onnx::OfflineASREngine engine(engine_config);

  const auto end_init = std::chrono::steady_clock::now();
  float elapsed_seconds_init =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_init -
                                                            begin_init)
          .count() /
      1000.0f;
  fprintf(stderr, "Engine created in %.3f s\n", elapsed_seconds_init);

  // 启动引擎
  fprintf(stderr, "Starting engine...\n");
  sherpa_onnx::ErrorCode code = sherpa_onnx::kSuccess;
  engine.Start(code);

  if (code != sherpa_onnx::kSuccess) {
    fprintf(stderr, "Failed to start engine. Error code: %d\n", code);
    return -1;
  }

  fprintf(stderr, "Engine started successfully!\n");

  // 处理音频文件
  fprintf(stderr, "\nProcessing audio files...\n");
  const auto begin = std::chrono::steady_clock::now();
  float total_duration = 0.0f;

  for (int32_t i = 1; i <= po.NumArgs(); ++i) {
    std::string wav_filename = po.GetArg(i);

    // 读取音频文件
    fprintf(stderr, "\n[%d/%d] Processing: %s\n", i, po.NumArgs(),
            wav_filename.c_str());

    int32_t sampling_rate = -1;
    bool is_ok = false;
    std::vector<float> samples =
        sherpa_onnx::ReadWave(wav_filename, &sampling_rate, &is_ok);

    if (!is_ok) {
      fprintf(stderr, "Failed to read '%s'\n", wav_filename.c_str());
      continue;
    }

    float duration = samples.size() / static_cast<float>(sampling_rate);
    total_duration += duration;
    fprintf(stderr, "  Audio duration: %.2f seconds\n", duration);
    fprintf(stderr, "  Sample rate: %d Hz\n", sampling_rate);
    fprintf(stderr, "  Number of samples: %zu\n", samples.size());

    // 创建会话
    fprintf(stderr, "  Creating session...\n");
    sherpa_onnx::ErrorCode session_code = sherpa_onnx::kSuccess;
    sherpa_onnx::OfflineSession *session = engine.CreateSession(session_code);

    if (session_code != sherpa_onnx::kSuccess || !session) {
      fprintf(stderr, "  Failed to create session. Error code: %d\n",
              session_code);
      continue;
    }

    // 接受音频数据
    fprintf(stderr, "  Accepting waveform...\n");
    session->AcceptWaveform(sampling_rate, samples.data(), samples.size());

    // 关闭会话（提交处理请求）
    fprintf(stderr, "  Closing session...\n");
    session->Close();

    // 注意: 结果获取机制可能需要根据实际实现进行调整
    // 当前接口可能使用回调或其他方式返回结果
    // 这里假设结果会在 Close() 后异步处理

    fprintf(stderr, "  Session completed.\n");

    // 注意: 根据实际实现，可能需要在这里等待结果
    // 或者使用回调机制获取识别结果
  }

  const auto end = std::chrono::steady_clock::now();
  float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.0f;

  fprintf(stderr, "\n========================================\n");
  fprintf(stderr, "Processing completed!\n");
  fprintf(stderr, "Total audio duration: %.2f seconds\n", total_duration);
  fprintf(stderr, "Elapsed time: %.2f seconds\n", elapsed_seconds);

  if (total_duration > 0) {
    float rtf = elapsed_seconds / total_duration;
    fprintf(stderr, "Real Time Factor (RTF): %.3f\n", rtf);
  }

  fprintf(stderr, "\nShutting down engine...\n");
  engine.Shutdown();
  fprintf(stderr, "Engine shutdown complete.\n");

  return 0;
}
