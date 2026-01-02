#include "sherpa-onnx/csrc/engine/offline-asr-engine/offline-asr-engine-config.h"

#include <sstream>

namespace sherpa_onnx {
std::string OfflineASREngineConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineASREngineConfig(";
  os << "num_workers=" << num_workers << ", ";
  os << "enable_gpu=" << (enable_gpu ? "True" : "False") << ", ";
  os << "use_vad=" << (use_vad ? "True" : "False") << ", ";
  os << "vad_config=" << vad_config.ToString() << ", ";
  os << "recognizer_config=" << recognizer_config.ToString() << ")";

  return os.str();
}

}  // namespace sherpa_onnx
