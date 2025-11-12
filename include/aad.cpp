#include "number.hpp"

Tape global_tape;

thread_local Tape* Number::tape = &global_tape;