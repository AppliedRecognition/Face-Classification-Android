#pragma once

#include <dlib/revision.h>
#if (DLIB_MAJOR_VERSION > 19) || (DLIB_MINOR_VERSION >= 11)
#include <dlib/cuda/tensor.h>  // moved here in 19.11
#else
#include <dlib/dnn/tensor.h>   // was here in 19.10 and earlier
#endif
#include <iterator>

namespace dlibx {
    extern const dlib::resizable_tensor empty_tensor; // in dnn_lmcon.cpp
}
