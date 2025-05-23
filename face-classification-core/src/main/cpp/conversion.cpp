#include "conversion.h"
#include <cassert>
#include <det/landmark_standardize.hpp>
#include <det/types.hpp>
#include <render/frontalize.hpp>
#include <iostream>
#include <stdext/binary.hpp>
#include <stdext/base64.hpp>
#include <json/json.hpp>
#include <json/types.hpp>
#include <fstream>

namespace verid {

    json::value toQuarter(float v) {
        const auto i = stdx::round_to<long>(4*v);
        if (i&3) return double(i) / 4.0;
        return i/4;
    }

    det::face_coordinates faceCoordinatesFromFace(JNIEnv *env, jobject face) {
        jclass faceClass = env->GetObjectClass(face);
        jfieldID leftEyeField = env->GetFieldID(faceClass, "leftEye", "Landroid/graphics/PointF;");
        jfieldID rightEyeField = env->GetFieldID(faceClass, "rightEye", "Landroid/graphics/PointF;");
        jfieldID landmarksField = env->GetFieldID(faceClass, "landmarks", "[Landroid/graphics/PointF;");
        jclass pointFCls = env->FindClass("android/graphics/PointF");
        jfieldID xField = env->GetFieldID(pointFCls, "x", "F");
        jfieldID yField = env->GetFieldID(pointFCls, "y", "F");
        jobject leftEyeObj = env->GetObjectField(face, leftEyeField);
        jobject rightEyeObj = env->GetObjectField(face, rightEyeField);
        float leftEyeX = env->GetFloatField(leftEyeObj, xField);
        float leftEyeY = env->GetFloatField(leftEyeObj, yField);
        float rightEyeX = env->GetFloatField(rightEyeObj, xField);
        float rightEyeY = env->GetFloatField(rightEyeObj, yField);
        jobjectArray landmarksArray = (jobjectArray)env->GetObjectField(face, landmarksField);
        jsize landmarkCount = env->GetArrayLength(landmarksArray);

        json::array arr;
        arr.reserve(1);
        json::array landmarks;
        if (landmarkCount == 478) {
            std::vector<int> indices = {
                    // jaw
                    127, 234, 93, 58, 172, 136, 149, 148, 152,
                    377, 378, 365, 397, 288, 323, 454, 356,
                    // eyebrows
                    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
                    // nose
                    168, 197, 195, 4, 240, 97, 2, 326, 460,
                    // eyes
                    33, 160, 158, 155, 153, 144, 382, 385, 387, 263, 373, 380,
                    // mouth (outer)
                    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
                    // mouth (inner)
                    78, 82, 13, 312, 308, 317, 14, 87
            };
            landmarks.reserve(indices.size());
            for (int i : indices) {
                jobject pointObj = env->GetObjectArrayElement(landmarksArray, i);
                float x = env->GetFloatField(pointObj, xField);
                float y = env->GetFloatField(pointObj, yField);
                json::array landmark = {toQuarter(x), toQuarter(y)};
                landmarks.push_back(landmark);
                env->DeleteLocalRef(pointObj);
            }
        } else {
            landmarks.reserve(landmarkCount);
            for (jsize i = 0; i < landmarkCount; ++i) {
                jobject pointObj = env->GetObjectArrayElement(landmarksArray, i);
                float x = env->GetFloatField(pointObj, xField);
                float y = env->GetFloatField(pointObj, yField);
                json::array landmark = {toQuarter(x), toQuarter(y)};
                landmarks.push_back(landmark);
                env->DeleteLocalRef(pointObj);
            }
        }
        json::object faceCoordinates = {
                {"t", "mesh68"},
                {"c", 10.0},
                {"el", json::array{toQuarter(leftEyeX), toQuarter(leftEyeY)}},
                {"er", json::array{toQuarter(rightEyeX), toQuarter(rightEyeY)}},
                {"lm", landmarks}
        };
        arr.push_back(move(faceCoordinates));
        env->DeleteLocalRef(pointFCls);
        env->DeleteLocalRef(faceClass);
        return det::face_coordinates(arr);
    }

    raw_image::plane rawImageFromImageObject(JNIEnv *env, jobject image) {
        raw_image::plane rawImage;

        jclass imageCls = env->GetObjectClass(image);
        jmethodID getWidth = env->GetMethodID(imageCls, "getWidth", "()I");
        jmethodID getHeight = env->GetMethodID(imageCls, "getHeight", "()I");
        jmethodID getBytesPerRow = env->GetMethodID(imageCls, "getBytesPerRow", "()I");
        rawImage.width = env->CallIntMethod(image, getWidth);
        rawImage.height = env->CallIntMethod(image, getHeight);
        rawImage.bytes_per_line = env->CallIntMethod(image, getBytesPerRow);

        jmethodID getData = env->GetMethodID(imageCls, "getData", "()[B");
        jbyteArray dataArray = (jbyteArray) env->CallObjectMethod(image, getData);
        jsize dataLen = env->GetArrayLength(dataArray);
        jbyte *dataPtr = env->GetByteArrayElements(dataArray, nullptr);

        jmethodID getFormat = env->GetMethodID(imageCls, "getFormat",
                                               "()Lcom/appliedrec/verid3/common/ImageFormat;");
        jobject formatObj = env->CallObjectMethod(image, getFormat);

        jclass formatCls = env->GetObjectClass(formatObj);
        jmethodID ordinalMethod = env->GetMethodID(formatCls, "ordinal", "()I");
        int formatOrdinal = env->CallIntMethod(formatObj, ordinalMethod);

        switch (formatOrdinal) {
            case 0: /* RGB */
                rawImage.layout = raw_image::pixel::rgb24;
                break;
            case 1: /* BGR */
                rawImage.layout = raw_image::pixel::bgr24;
                break;
            case 2: /* ARGB */
                rawImage.layout = raw_image::pixel::argb32;
                break;
            case 3: /* BGRA */
                rawImage.layout = raw_image::pixel::bgra32;
                break;
            case 4: /* ABGR */
                rawImage.layout = raw_image::pixel::abgr32;
                break;
            case 5: /* RGBA */
                rawImage.layout = raw_image::pixel::rgba32;
                break;
            case 6: /* GRAYSCALE */
                rawImage.layout = raw_image::pixel::gray8;
                break;
            default: /* Unknown */
                throw std::invalid_argument("unsupported image format");
        }
        rawImage.data = new unsigned char[dataLen];
        std::memcpy(rawImage.data, dataPtr, dataLen);
        env->ReleaseByteArrayElements(dataArray, dataPtr, JNI_ABORT);
        env->DeleteLocalRef(formatObj);
        env->DeleteLocalRef(formatCls);
        env->DeleteLocalRef(imageCls);
        return rawImage;
    }
}