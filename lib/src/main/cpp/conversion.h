#pragma once

#include <memory>
#include <filesystem>
#include <stdext/binary.hpp>
#include <det/types.hpp>
#include "jni.h"

namespace verid {
    det::face_coordinates faceCoordinatesFromFace(JNIEnv  *env, jobject face);
    raw_image::plane rawImageFromImageObject(JNIEnv *env, jobject image);
}