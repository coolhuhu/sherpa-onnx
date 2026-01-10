#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-asr-engine-config.h"

#include <sstream>

namespace sherpa_onnx {
std::string OfflineASREngineConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineASREngineConfig(";
  os << "num_worker_threads=" << num_worker_threads << ", ";
  os << "enable_vad=" << (enable_vad ? "True" : "False") << ", ";
  os << "max_accept_waveform_size=" << max_accept_waveform_size << ", ";
  os << "max_batch_size=" << max_batch_size << ", ";
  os << "enable_task_stealing=" << (enable_task_stealing ? "True" : "False")
     << ", ";
  os << "enable_gpu=" << (enable_gpu ? "True" : "False") << ", ";
  os << "vad_config=" << vad_config.ToString() << ", ";
  os << "recognizer_config=" << recognizer_config.ToString() << ")";

  return os.str();
}

}  // namespace sherpa_onnx
