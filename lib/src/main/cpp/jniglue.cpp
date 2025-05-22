#include <jni.h>
#include "FaceClassifier.h"
#include <memory>
#include "conversion.h"
#include "raw_image/types.hpp"

extern "C"
JNIEXPORT void JNICALL
Java_com_appliedrec_verid3_faceclassification_FaceClassifier_destroyNativeContext(JNIEnv *env,
                                                                                  jobject thiz,
                                                                                  jlong context) {
    auto *faceClassifier = reinterpret_cast<verid::FaceClassifier *>(context);
    delete faceClassifier;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_appliedrec_verid3_faceclassification_FaceClassifier_extractAttribute(JNIEnv *env,
                                                                              jobject thiz,
                                                                              jlong context,
                                                                              jobject face,
                                                                              jobject image) {
    try {
        auto *faceClassifier = reinterpret_cast<verid::FaceClassifier *>(context);
        if (!faceClassifier) {
            throw std::exception();
        }
        auto faceCoords = verid::faceCoordinatesFromFace(env, face);
        auto rawImage = verid::rawImageFromImageObject(env, image);
        std::vector<float> result = faceClassifier->extractAttribute(faceCoords, rawImage);
        auto resultLength = static_cast<jsize>(result.size());
        jfloatArray resultArray = env->NewFloatArray(resultLength);
        if (resultLength > 0) {
            env->SetFloatArrayRegion(resultArray, 0, resultLength, result.data());
        }
        return resultArray;
    } catch (std::exception &e) {
        jclass exceptionClass = env->FindClass("java/lang/Exception");
        env->ThrowNew(exceptionClass, e.what());
        return nullptr;
    } catch (const char* msg) {
        jclass exceptionClass = env->FindClass("java/lang/Exception");
        env->ThrowNew(exceptionClass, msg);
    } catch (int code) {
        std::string msg = "Classification exception with code: " + std::to_string(code);
        jclass exceptionClass = env->FindClass("java/lang/Exception");
        env->ThrowNew(exceptionClass, msg.c_str());
    } catch (...) {
        jclass exceptionClass = env->FindClass("java/lang/Exception");
        env->ThrowNew(exceptionClass, "Classification failed");
        return nullptr;
    }
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_appliedrec_verid3_faceclassification_FaceClassifier_createNativeContext(JNIEnv *env,
                                                                                 jclass clazz,
                                                                                 jstring name,
                                                                                 jbyteArray modelBytes) {
    try {
        const char *chars = env->GetStringUTFChars(name, nullptr);
        std::string classifierName(chars);
        env->ReleaseStringUTFChars(name, chars);
        jsize len = env->GetArrayLength(modelBytes);
        jbyte *bytes = env->GetByteArrayElements(modelBytes, nullptr);
        std::vector<unsigned char> buf(len);
        if (len > 0) {
            std::memcpy(buf.data(), bytes, len);
        }
        auto modelBuffer = std::make_unique<stdx::binary>(buf);
        env->ReleaseByteArrayElements(modelBytes, bytes, JNI_ABORT);
        auto *faceClassifier = new verid::FaceClassifier(classifierName, std::move(modelBuffer));
        return reinterpret_cast<jlong>(faceClassifier);
    } catch (...) {
        jclass exceptionClass = env->FindClass("java/lang/Exception");
        env->ThrowNew(exceptionClass, "Classification failed");
        return -1;
    }
}