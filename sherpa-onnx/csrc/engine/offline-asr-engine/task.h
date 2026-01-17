#ifndef OFFLINE_ASR_ENGINE_TASK_H_
#define OFFLINE_ASR_ENGINE_TASK_H_

#include <vector>

#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-session-impl.h"

namespace sherpa_onnx {

struct WaveTask {
  int32_t task_id;
  int32_t sample_rate;
  int32_t start;
  std::vector<float> samples;

  OfflineSessionImpl *session;

  WaveTask() = default;

  WaveTask(int32_t task_id, int32_t sample_rate, int32_t start,
           const float *samples, int32_t n, OfflineSessionImpl *session)
      : task_id(task_id),
        sample_rate(sample_rate),
        start(start),
        samples(samples, samples + n),
        session(session) {}

  WaveTask(WaveTask &&rhs)
      : task_id(rhs.task_id),
        sample_rate(rhs.sample_rate),
        start(rhs.start),
        samples(std::move(rhs.samples)),
        session(rhs.session) {}

  WaveTask &operator=(WaveTask &&rhs) {
    std::swap(task_id, rhs.task_id);
    std::swap(sample_rate, rhs.sample_rate);
    std::swap(start, rhs.start);
    std::swap(samples, rhs.samples);
    std::swap(session, rhs.session);
    return *this;
  }
};

struct SegmentTask {
  int32_t segment_id;
  float start_time;
  float end_time;
  std::unique_ptr<OfflineStream> stream;
  OfflineSessionImpl *session;

  SegmentTask() = default;

  SegmentTask(SegmentTask &&rhs)
      : segment_id(rhs.segment_id),
        start_time(rhs.start_time),
        end_time(rhs.end_time),
        stream(std::move(rhs.stream)),
        session(rhs.session) {}

  SegmentTask &operator=(SegmentTask &&rhs) {
    std::swap(segment_id, rhs.segment_id);
    std::swap(start_time, rhs.start_time);
    std::swap(end_time, rhs.end_time);
    std::swap(stream, rhs.stream);
    std::swap(session, rhs.session);
    return *this;
  }
};

}  // namespace sherpa_onnx

#endif  // OFFLINE_ASR_ENGINE_TASK_H_