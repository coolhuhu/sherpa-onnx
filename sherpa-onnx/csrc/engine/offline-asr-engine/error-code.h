#ifndef OFFLINE_ASR_ENGINE_ERROR_CODE_H_
#define OFFLINE_ASR_ENGINE_ERROR_CODE_H_

#include <string>

struct ErrorCode {
  enum Code {
    kSuccess = 0,
    kResourceExhausted = 1,
    kInvalidArgument = 2,
    kEngineUninitialized = 3
  };

  Code error_code = kSuccess;

  std::string error_msg;
};

#endif  // OFFLINE_ASR_ENGINE_ERROR_CODE_H_