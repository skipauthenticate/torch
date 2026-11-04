// The one exception type of the engine.
#pragma once

#include <stdexcept>

namespace torch {

// Thrown by loading and setup, never by the decode loop. The message is one sentence that says what went wrong and
// what to do about it.
class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

}  // namespace torch
