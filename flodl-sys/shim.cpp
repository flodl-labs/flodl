// shim.cpp — unity-build entry point.
//
// torch/torch.h is enormous and dominates per-TU compile cost (~17s alone).
// Splitting the FFI surface across multiple TUs would multiply that cost
// (cc::Build does not cache per-file, so any edit triggers all-file rebuilds
// — measured as ~17s × N).
//
// We get the best of both worlds with a unity build: developers navigate
// topic-focused files (ops_tensor.cpp, ops_nn.cpp, etc.), but the compiler
// sees a single translation unit and parses torch.h exactly once.
//
// Each included file is self-contained but starts with `#include "helpers.h"`,
// which is `#pragma once`-guarded. Translation-unit-local `static` helpers
// inside each file remain file-private to this aggregate TU.

#include "ops_tensor.cpp"
#include "ops_nn.cpp"
#include "ops_math_ext.cpp"
#include "ops_training.cpp"
#include "ops_cuda.cpp"
