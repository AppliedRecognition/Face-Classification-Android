
#include "png.hpp"

#include <stdexcept>
#include <applog/core.hpp>

using namespace raw_image;

std::unique_ptr<reader>
raw_image::png_load(stdx::file_ptr, pixel_layout) {
    FILE_LOG(logWARNING) << "png not supported";
    throw std::runtime_error("libpng not available");
}

std::unique_ptr<reader>
raw_image::png_load(stdx::binary data, pixel_layout) {
    FILE_LOG(logWARNING) << "png not supported (" << data.size() << " bytes)";
    throw std::runtime_error("libpng not available");
}

std::unique_ptr<reader>
raw_image::png_load(const void*, std::size_t size, pixel_layout) {
    FILE_LOG(logWARNING) << "png not supported (" << size << " bytes)";
    throw std::runtime_error("libpng not available");
}

stdx::binary internal::png_binary(plane const*) {
    FILE_LOG(logWARNING) << "png not supported";
    throw std::runtime_error("libpng not available");
}
