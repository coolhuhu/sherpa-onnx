// sherpa-onnx/csrc/online-wav2vec-audio.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_WAV2VEC_AUDIO_H_
#define SHERPA_ONNX_CSRC_ONLINE_WAV2VEC_AUDIO_H_

#include <cstdint>
#include <mutex>
#include <vector>

namespace sherpa_onnx {

/**
 * Configuration for wav2vec2.0 raw audio feature extraction.
 *
 * Wav2vec2.0 uses a CNN frontend with the following characteristics:
 * - chunk_length: Frame length in samples (default: 400 samples = 25ms @ 16kHz)
 * - chunk_shift: Frame shift in samples (default: 320 samples = 20ms @ 16kHz)
 * - Overlap between frames: chunk_length - chunk_shift = 80 samples
 *
 * For streaming with 32 frames:
 * Total samples = (32 - 1) * chunk_shift + chunk_length
 *              = 31 * 320 + 400 = 10320 samples @ 16kHz
 */
struct OnlineWav2VecAudioConfig {
  // Sampling rate (typically 16000 for wav2vec2.0)
  int32_t sampling_rate = 16000;

  // Frame length in milliseconds (typically 25ms)
  float chunk_length_ms = 25.0f;

  // Frame shift in milliseconds (typically 20ms)
  float chunk_shift_ms = 20.0f;

  // Calculate derived parameters
  int32_t ChunkLength() const {
    return static_cast<int32_t>(chunk_length_ms * sampling_rate / 1000);
  }

  int32_t ChunkShift() const {
    return static_cast<int32_t>(chunk_shift_ms * sampling_rate / 1000);
  }

  // Calculate total samples needed for n frames
  // Formula: (n - 1) * chunk_shift + chunk_length
  int32_t GetNumSamples(int32_t n_frames) const {
    return (n_frames - 1) * ChunkShift() + ChunkLength();
  }

  std::string ToString() const;
};

/**
 * This class implements raw audio feature extraction for wav2vec2.0 models.
 *
 * Unlike traditional feature extractors (fbank, mfcc) that extract
 * spectral features per frame, wav2vec2.0 works with overlapping audio frames.
 *
 * Key differences from traditional feature extraction:
 * 1. No spectral transformation - returns raw audio samples
 * 2. Frames have overlap (chunk_length > chunk_shift)
 * 3. FeatureDim() returns chunk_length (number of samples per frame)
 * 4. GetFrames(n) returns (n-1)*chunk_shift + chunk_length samples
 *
 * Usage example for streaming (32 frames @ 16kHz):
 *   OnlineWav2VecAudio audio;
 *   audio.AcceptWaveform(16000, samples, n);
 *   int num_frames = audio.NumFramesReady();
 *   std::vector<float> chunk = audio.GetFrames(0, 32);  // Returns 10320 samples
 *
 * The returned samples can be directly fed to wav2vec2.0 model,
 * which will process them through its 7-layer CNN frontend.
 */
class OnlineWav2VecAudio {
 public:
  explicit OnlineWav2VecAudio(const OnlineWav2VecAudioConfig &config);
  ~OnlineWav2VecAudio() = default;

  /**
   * Accept waveform samples and cache them internally.
   *
   * @param sampling_rate Sampling rate of the input waveform (e.g., 16000)
   * @param waveform Pointer to audio samples, should be normalized to [-1, 1]
   * @param n Number of samples
   */
  void AcceptWaveform(int32_t sampling_rate, const float *waveform, int32_t n);

  /**
   * Signal that no more waveform will be provided.
   */
  void InputFinished();

  /**
   * Get the number of frames that can be extracted.
   * Formula: floor((num_samples - chunk_length) / chunk_shift) + 1
   * Returns 0 if not enough samples for a full frame.
   *
   * @return Number of extractable frames
   */
  int32_t NumFramesReady() const;

  /**
   * Check if a given frame is the last one.
   *
   * @param frame Index of the frame
   * @return True if this is the last frame and InputFinished() was called
   */
  bool IsLastFrame(int32_t frame) const;

  /**
   * Get n frames starting from the given frame index.
   * The returned vector contains overlapping audio samples.
   *
   * Total samples = (n - 1) * chunk_shift + chunk_length
   *
   * Example: For 32 frames with chunk_length=400, chunk_shift=320
   * Returns: (32-1)*320 + 400 = 10320 samples
   *
   * @param frame_index Starting frame index
   * @param n Number of frames to retrieve
   * @return Vector of audio samples with overlaps
   */
  std::vector<float> GetFrames(int32_t frame_index, int32_t n) const;

  /**
   * Discard the first n frames from the cache.
   *
   * @param n Number of frames to discard
   */
  void Pop(int32_t n);

  /**
   * Get the feature dimension.
   * Returns the number of samples in one frame (chunk_length).
   *
   * @return Feature dimension (chunk_length in samples)
   */
  int32_t Dim() const { return config_.ChunkLength(); }

  /**
   * Get the sampling rate.
   */
  int32_t GetSamplingRate() const { return config_.sampling_rate; }

 private:
  int32_t GetFrameStartSample(int32_t frame_index) const;

  OnlineWav2VecAudioConfig config_;
  mutable std::mutex mutex_;
  std::vector<float> samples_;  // Cached audio samples
  bool input_finished_ = false;
  mutable int32_t last_frame_index_ = 0;  // Track last accessed frame for Pop operations
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_WAV2VEC_AUDIO_H_
