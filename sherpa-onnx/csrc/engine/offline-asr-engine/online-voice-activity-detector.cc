#include "sherpa-onnx/csrc/engine/offline-asr-engine/online-voice-activity-detector.h"

#include <cassert>
#include <iostream>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/voice-activity-detector.h"

namespace sherpa_onnx {

class OnlineVoiceActivityDetector::Impl {
 public:
  Impl(const VadModelConfig &config)
      : vad_detector_(config),
        num_processed_samples_(0),
        speech_segment_start_(false) {
    if (!config.silero_vad.model.empty()) {
      window_size_ = config.silero_vad.window_size;
    } else if (!config.ten_vad.model.empty()) {
      window_size_ = config.ten_vad.window_size;
    } else {
      SHERPA_ONNX_LOGE("Currently, we only support Silero VAD or Ten VAD.");
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void AcceptWaveform(const float *samples, int32_t n) {
    if (n <= 0) {
      return;
    }

    int32_t offset = 0;
    while (offset + window_size_ < n) {
      vad_detector_.AcceptWaveform(samples + offset, window_size_);

      offset += window_size_;

      // Detect the start of the speech segment
      if (!speech_segment_start_ && vad_detector_.IsSpeechDetected()) {
        speech_segment_start_ = true;
        auto s = vad_detector_.CurrentSpeechSegment();
        num_processed_samples_ = s.start;
      }

      // detected the endpoint
      while (!vad_detector_.Empty()) {
        auto s = vad_detector_.Front();
        vad_detector_.Pop();

        if (num_processed_samples_ <
            (s.start + static_cast<int32_t>(s.samples.size()))) {
          int32_t tail_chunk_size = s.start +
                                    static_cast<int32_t>(s.samples.size()) -
                                    num_processed_samples_;

          VadSpeechSegment segment;
          segment.start = num_processed_samples_;
          segment.samples = std::vector<float>(
              s.samples.end() - tail_chunk_size, s.samples.end());
          segment.endpoint = true;
          speech_segments_.push_back(std::move(segment));
          num_processed_samples_ = s.start + s.samples.size();
        } else {
          VadSpeechSegment segment;
          segment.endpoint = true;
          speech_segments_.push_back(std::move(segment));
        }

        speech_segment_start_ = false;
      }
    }

    // process audio stream that smaller than one window_size
    vad_detector_.AcceptWaveform(samples + offset, n - offset);

    if (vad_detector_.IsSpeechDetected()) {
      if (!speech_segment_start_) {
        speech_segment_start_ = true;
        auto s = vad_detector_.CurrentSpeechSegment();
        num_processed_samples_ = s.start;
      }

      auto s = vad_detector_.CurrentSpeechSegment();
      int32_t tail_size = s.start + s.samples.size() - num_processed_samples_;
      VadSpeechSegment segment;
      segment.start = num_processed_samples_;
      segment.samples =
          std::vector<float>(s.samples.end() - tail_size, s.samples.end());
      speech_segments_.push_back(std::move(segment));
      num_processed_samples_ = s.start + s.samples.size();
    }
  }

  bool Empty() const { return speech_segments_.empty(); }

  void PopAll() { speech_segments_.clear(); }

  void Flush() const { vad_detector_.Flush(); }

  std::vector<VadSpeechSegment> GetSpeechSegment() {
    return std::move(speech_segments_);
  }

  void Reset() {
    vad_detector_.Reset();
    speech_segments_.clear();
    num_processed_samples_ = 0;
    speech_segment_start_ = false;
  }

 private:
  int32_t window_size_;
  VoiceActivityDetector vad_detector_;
  int32_t num_processed_samples_;
  bool speech_segment_start_;
  std::vector<VadSpeechSegment> speech_segments_;
};

OnlineVoiceActivityDetector::OnlineVoiceActivityDetector(
    const VadModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OnlineVoiceActivityDetector::~OnlineVoiceActivityDetector() = default;

void OnlineVoiceActivityDetector::AcceptWaveform(const float *samples,
                                                 int32_t n) {
  impl_->AcceptWaveform(samples, n);
}

void OnlineVoiceActivityDetector::AcceptWaveform(
    const std::vector<float> &samples) {
  AcceptWaveform(samples.data(), samples.size());
}

void OnlineVoiceActivityDetector::Reset() { impl_->Reset(); }

bool OnlineVoiceActivityDetector::Empty() const { return impl_->Empty(); }

void OnlineVoiceActivityDetector::PopAll() { impl_->PopAll(); }

std::vector<VadSpeechSegment> OnlineVoiceActivityDetector::GetSpeechSegment() {
  return impl_->GetSpeechSegment();
}

void OnlineVoiceActivityDetector::Flush() const { impl_->Flush(); }

}  // namespace sherpa_onnx
