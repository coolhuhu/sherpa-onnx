#include <algorithm>
#include <iostream>
#include <string>

#include "sherpa-onnx/csrc/features.h"
#include "sherpa-onnx/csrc/wave-reader.h"

using namespace sherpa_onnx;

int main(int argc, char **argv) {
  std::string wav_path = argv[1];

  int32_t sampling_rate;
  bool is_ok;
  std::vector<float> samples = ReadWave(wav_path, &sampling_rate, &is_ok);

  if (!is_ok) {
    std::cerr << "Failed to read " << wav_path << std::endl;
    exit(-1);
  }

  FeatureExtractorConfig config;
  config.sampling_rate = 16000;
  config.is_t_one = true;
  config.snip_edges = true;
  config.frame_length_ms = 25.0f;
  config.frame_shift_ms = 20.0f;
  config.round_to_power_of_two = false;

  int32_t chunk_length = 10320;
  FeatureExtractor feature_extractor(config);
  std::cout << "feature dim: " << feature_extractor.FeatureDim() << std::endl;

  feature_extractor.AcceptWaveform(sampling_rate, samples.data(), chunk_length);
  int32_t num_ready_frames = feature_extractor.NumFramesReady();
  std::cout << "num frames ready: " << num_ready_frames << std::endl;

  std::vector<float> frames = feature_extractor.GetFrames(0, num_ready_frames);

  std::cout << "frames.size(): " << frames.size() << std::endl;

  std::for_each(frames.begin() + chunk_length, frames.end(),
                [](float x) { std::cout << x << " "; });
  std::cout << std::endl;
}