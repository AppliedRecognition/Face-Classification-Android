
#include "tiff.hpp"

#include <stdexcept>
#include <applog/core.hpp>

using namespace raw_image;

plane_ptr raw_image::tiff_load(const void*, std::size_t size) {
    FILE_LOG(logWARNING) << "tiff not supported (" << size << " bytes)";
    throw std::runtime_error("libtiff not available");
}

plane_ptr raw_image::tiff_load(FILE*) {
    FILE_LOG(logWARNING) << "tiff not supported";
    throw std::runtime_error("libtiff not available");
}
