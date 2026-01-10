#ifndef OFFLINE_ASR_ENGINE_TASK_H_
#define OFFLINE_ASR_ENGINE_TASK_H_

#include <vector>

#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-session-impl.h"

namespace sherpa_onnx {

struct WaveTask {
  int32_t sample_rate;
  int32_t start;
  std::vector<float> samples;

  OfflineSessionImpl *session;

  WaveTask() = default;

  WaveTask(int32_t sample_rate, int32_t start, const float *samples, int32_t n,
           OfflineSessionImpl *session)
      : sample_rate(sample_rate),
        start(start),
        samples(samples, samples + n),
        session(session) {}

  WaveTask(WaveTask &&rhs)
      : sample_rate(rhs.sample_rate),
        start(rhs.start),
        samples(std::move(rhs.samples)),
        session(rhs.session) {}

  WaveTask &operator=(WaveTask &&rhs) {
    std::swap(sample_rate, rhs.sample_rate);
    std::swap(start, rhs.start);
    std::swap(samples, rhs.samples);
    std::swap(session, rhs.session);
    return *this;
  }
};

}  // namespace sherpa_onnx

#endif  // OFFLINE_ASR_ENGINE_TASK_H_