// sherpa-onnx/csrc/online-wav2vec-audio.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-wav2vec-audio.h"

#include <algorithm>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

std::string OnlineWav2VecAudioConfig::ToString() const {
  std::ostringstream os;
  os << "OnlineWav2VecAudioConfig(";
  os << "sampling_rate=" << sampling_rate << ", ";
  os << "chunk_length_ms=" << chunk_length_ms << ", ";
  os << "chunk_shift_ms=" << chunk_shift_ms << ", ";
  os << "chunk_length=" << ChunkLength() << " samples, ";
  os << "chunk_shift=" << ChunkShift() << " samples)";
  return os.str();
}

OnlineWav2VecAudio::OnlineWav2VecAudio(const OnlineWav2VecAudioConfig &config)
    : config_(config) {
  // Validate configuration
  if (config_.sampling_rate <= 0) {
    SHERPA_ONNX_LOGE("Invalid sampling_rate: %d", config_.sampling_rate);
    exit(-1);
  }
  if (config_.chunk_length_ms <= 0 || config_.chunk_shift_ms <= 0) {
    SHERPA_ONNX_LOGE("Invalid chunk_length_ms or chunk_shift_ms");
    exit(-1);
  }
  if (config_.ChunkLength() <= config_.ChunkShift()) {
    SHERPA_ONNX_LOGE(
        "chunk_length must be greater than chunk_shift: %d vs %d",
        config_.ChunkLength(), config_.ChunkShift());
    exit(-1);
  }
}

void OnlineWav2VecAudio::AcceptWaveform(int32_t sampling_rate,
                                       const float *waveform, int32_t n) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Handle resampling if needed
  if (sampling_rate != config_.sampling_rate) {
    SHERPA_ONNX_LOGE(
        "Sampling rate mismatch: expected %d, got %d. "
        "Resampling is not supported yet. Please provide audio at %d Hz.",
        config_.sampling_rate, sampling_rate, config_.sampling_rate);
    exit(-1);
  }

  // Append samples to internal buffer
  samples_.insert(samples_.end(), waveform, waveform + n);
}

void OnlineWav2VecAudio::InputFinished() {
  std::lock_guard<std::mutex> lock(mutex_);
  input_finished_ = true;
}

int32_t OnlineWav2VecAudio::NumFramesReady() const {
  std::lock_guard<std::mutex> lock(mutex_);

  int32_t chunk_length = config_.ChunkLength();
  int32_t chunk_shift = config_.ChunkShift();

  if (samples_.size() < static_cast<size_t>(chunk_length)) {
    return 0;
  }

  // Number of frames = floor((num_samples - chunk_length) / chunk_shift) + 1
  int32_t num_frames =
      (static_cast<int32_t>(samples_.size()) - chunk_length) / chunk_shift + 1;

  return num_frames;
}

bool OnlineWav2VecAudio::IsLastFrame(int32_t frame) const {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!input_finished_) {
    return false;
  }

  return frame == NumFramesReady() - 1;
}

int32_t OnlineWav2VecAudio::GetFrameStartSample(int32_t frame_index) const {
  // Frame i starts at sample: frame_index * chunk_shift
  return frame_index * config_.ChunkShift();
}

std::vector<float> OnlineWav2VecAudio::GetFrames(int32_t frame_index,
                                                 int32_t n) const {
  std::lock_guard<std::mutex> lock(mutex_);

  int32_t num_frames_ready = NumFramesReady();

  if (frame_index < 0) {
    SHERPA_ONNX_LOGE("Invalid frame_index: %d (< 0)", frame_index);
    exit(-1);
  }

  if (n <= 0) {
    SHERPA_ONNX_LOGE("Invalid n: %d (must be > 0)", n);
    exit(-1);
  }

  if (frame_index + n > num_frames_ready) {
    SHERPA_ONNX_LOGE(
        "Cannot get %d frames starting from frame %d. Only %d frames ready.",
        n, frame_index, num_frames_ready);
    exit(-1);
  }

  // Check if we're trying to go backwards
  if (frame_index < last_frame_index_) {
    SHERPA_ONNX_LOGE(
        "Cannot get frames backwards. last_frame_index_: %d, frame_index: %d",
        last_frame_index_, frame_index);
    exit(-1);
  }

  // Discard frames that we're skipping
  int32_t discard_num = frame_index - last_frame_index_;
  if (discard_num > 0) {
    const_cast<OnlineWav2VecAudio *>(this)->Pop(discard_num);
  }

  // Calculate total samples needed
  // Formula: (n - 1) * chunk_shift + chunk_length
  int32_t chunk_shift = config_.ChunkShift();
  int32_t chunk_length = config_.ChunkLength();
  int32_t total_samples = (n - 1) * chunk_shift + chunk_length;

  // Calculate starting sample position
  int32_t start_sample = frame_index * chunk_shift;

  // Extract samples
  std::vector<float> result(total_samples);

  for (int32_t i = 0; i < total_samples; ++i) {
    result[i] = samples_[start_sample + i];
  }

  // Update last_frame_index_
  last_frame_index_ = frame_index;

  return result;
}

void OnlineWav2VecAudio::Pop(int32_t n) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (n <= 0) {
    return;
  }

  int32_t chunk_shift = config_.ChunkShift();
  int32_t samples_to_remove = n * chunk_shift;

  if (static_cast<size_t>(samples_to_remove) > samples_.size()) {
    SHERPA_ONNX_LOGE(
        "Cannot pop %d frames (%d samples). Only %zu samples in buffer.",
        n, samples_to_remove, samples_.size());
    exit(-1);
  }

  // Remove samples from the beginning
  samples_.erase(samples_.begin(), samples_.begin() + samples_to_remove);

  // Update frame index tracking
  last_frame_index_ -= n;
  if (last_frame_index_ < 0) {
    last_frame_index_ = 0;
  }
}

}  // namespace sherpa_onnx
