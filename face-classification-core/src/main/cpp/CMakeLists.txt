cmake_minimum_required(VERSION 3.22.1)
project(FaceClassification)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set Android toolchain & target
if(ANDROID)
    message(STATUS "Building for Android")
    # You may need to specify ABI and API level outside of CMake, e.g. -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-21
    set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    add_definitions(-DANDROID)
else()
    message(FATAL_ERROR "This CMake file is for Android builds only")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -frtti -fexceptions -g")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -frtti -fexceptions")
set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/build)
set(LIB_INTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Lib-Internal)
set(LIB_EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Lib-External)

add_library(openblas STATIC IMPORTED)
set_target_properties(openblas PROPERTIES IMPORTED_LOCATION ${LIB_EXTERNAL_DIR}/openblasbuild/output/${CMAKE_ANDROID_ARCH_ABI}/libopenblas.a)
target_include_directories(openblas INTERFACE ${LIB_EXTERNAL_DIR}/openblasbuild/output/${CMAKE_ANDROID_ARCH_ABI}include)

include_directories(${LIB_EXTERNAL_DIR}/dlib-proper)
include_directories(${LIB_EXTERNAL_DIR}/boost_1_85_0)

##########
# APPLOG #
##########

add_library(applog STATIC
        ${LIB_INTERNAL_DIR}/applog/assert.cpp
        ${LIB_INTERNAL_DIR}/applog/internal.cpp
        ${LIB_INTERNAL_DIR}/applog/module.cpp
        ${LIB_INTERNAL_DIR}/applog/logger.cpp
        ${LIB_INTERNAL_DIR}/applog/sink.cpp
        ${LIB_INTERNAL_DIR}/applog/filter_sink.cpp
        ${LIB_INTERNAL_DIR}/applog/ostream_sink.cpp
)
set_target_properties(applog PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(applog PUBLIC APPLOG_MINIMUM_LEVEL=logNONE)
target_include_directories(applog PUBLIC ${LIB_EXTERNAL_DIR}/boost_1_85_0)

########
# CORE #
########

add_library(core STATIC
        ${LIB_INTERNAL_DIR}/core/context.cpp
        ${LIB_INTERNAL_DIR}/core/thread_data.cpp
)
set_target_properties(core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(core PRIVATE applog)
target_include_directories(core PRIVATE ${LIB_INTERNAL_DIR})

########
# JSON #
########

add_library(json STATIC
        ${LIB_INTERNAL_DIR}/json/types.cpp
        ${LIB_INTERNAL_DIR}/json/io_manip.cpp
        ${LIB_INTERNAL_DIR}/json/encode.cpp
        ${LIB_INTERNAL_DIR}/json/push_types.cpp
        ${LIB_INTERNAL_DIR}/json/pull_types.cpp
        ${LIB_INTERNAL_DIR}/json/pull_encode_json.cpp
        ${LIB_INTERNAL_DIR}/json/pull_encode_amf3.cpp
        ${LIB_INTERNAL_DIR}/json/push_decode_json.cpp
        ${LIB_INTERNAL_DIR}/json/push_decode_amf3.cpp
        ${LIB_INTERNAL_DIR}/json/pull_encode_cbor.cpp
        ${LIB_INTERNAL_DIR}/json/push_decode_cbor.cpp
        ${LIB_INTERNAL_DIR}/json/cbor_decode.cpp
        ${LIB_INTERNAL_DIR}/json/zlib.cpp
)
set_target_properties(json PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(json PRIVATE applog)
target_include_directories(json PRIVATE ${LIB_INTERNAL_DIR})

#############
# RAW_IMAGE #
#############

add_library(raw_image STATIC
        ${LIB_INTERNAL_DIR}/raw_image/reader.cpp
        ${LIB_INTERNAL_DIR}/raw_image/reader_scale.cpp
        ${LIB_INTERNAL_DIR}/raw_image/reader_rotate.cpp
        ${LIB_INTERNAL_DIR}/raw_image/core.cpp
        ${LIB_INTERNAL_DIR}/raw_image/transform.cpp
        ${LIB_INTERNAL_DIR}/raw_image/image_quality.cpp
        ${LIB_INTERNAL_DIR}/raw_image/linear_regression.cpp
        ${LIB_INTERNAL_DIR}/raw_image/scaled_chip.cpp
        ${LIB_INTERNAL_DIR}/raw_image/input_extractor.cpp
        ${LIB_INTERNAL_DIR}/raw_image/input_extractor_retina.cpp
        ${LIB_INTERNAL_DIR}/raw_image/adjust.cpp
        ${LIB_INTERNAL_DIR}/raw_image/drawing.cpp
        ${LIB_INTERNAL_DIR}/raw_image/face_landmarks.cpp
        ${LIB_INTERNAL_DIR}/raw_image/face_types.cpp
)
set_target_properties(raw_image PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(raw_image PRIVATE applog)
target_include_directories(raw_image PRIVATE ${LIB_INTERNAL_DIR})

###########
# MODELS  #
###########

add_library(models STATIC
        ${LIB_INTERNAL_DIR}/models/loader_std.cpp
)
set_target_properties(models PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(models PRIVATE ${LIB_INTERNAL_DIR})
target_link_libraries(models PRIVATE applog)

#######
# DET #
#######

add_library(det STATIC
        ${LIB_INTERNAL_DIR}/det/landmarks.cpp
        ${LIB_INTERNAL_DIR}/det/pose.cpp
        ${LIB_INTERNAL_DIR}/det/types.cpp
        ${LIB_INTERNAL_DIR}/det/detection_settings.cpp
        ${LIB_INTERNAL_DIR}/det/image.cpp
        ${LIB_INTERNAL_DIR}/det/detection.cpp
        ${LIB_INTERNAL_DIR}/det/drawing.cpp
)
set_target_properties(det PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(det PRIVATE core json raw_image models)
target_include_directories(det PRIVATE ${LIB_INTERNAL_DIR} ${LIB_EXTERNAL_DIR}/boost_1_85_0)

########
# DLIB #
########

add_library(dlib STATIC
        ${LIB_EXTERNAL_DIR}/dlib-proper/dlib/all/source.cpp
)
set_target_properties(dlib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(dlib PRIVATE DLIB_NO_GUI_SUPPORT=1)
target_include_directories(dlib PUBLIC ${LIB_EXTERNAL_DIR}/dlib-proper/dlib/image_keypoint)
target_link_libraries(dlib PRIVATE openblas z)

#########
# DLIBX #
#########

add_library(dlibx STATIC
        ${LIB_INTERNAL_DIR}/dlibx/shape_predictor.cpp
        ${LIB_INTERNAL_DIR}/dlibx/shape_extract_pixels.cpp
        ${LIB_INTERNAL_DIR}/dlibx/shape_quality_lbp.cpp
        ${LIB_INTERNAL_DIR}/dlibx/raw_image.cpp
        ${LIB_INTERNAL_DIR}/dlibx/hash32.cpp
        ${LIB_INTERNAL_DIR}/dlibx/input_extractor.cpp
        ${LIB_INTERNAL_DIR}/dlibx/input_extractor_facechip.cpp
        ${LIB_INTERNAL_DIR}/dlibx/net_layer.cpp
        ${LIB_INTERNAL_DIR}/dlibx/net_layer_impl.cpp
        ${LIB_INTERNAL_DIR}/dlibx/net_vector.cpp
        ${LIB_INTERNAL_DIR}/dlibx/net_convert.cpp
        ${LIB_INTERNAL_DIR}/dlibx/bfloat16.cpp
        ${LIB_INTERNAL_DIR}/dlibx/chip_details.cpp
        ${LIB_INTERNAL_DIR}/dlibx/linear_regression.cpp
        ${LIB_INTERNAL_DIR}/dlibx/library_init.cpp
        ${LIB_INTERNAL_DIR}/dlibx/qmat.cpp
        ${LIB_INTERNAL_DIR}/dlibx/conv_blas.cpp
        ${LIB_INTERNAL_DIR}/dlibx/tensor_conv.cpp
        ${LIB_INTERNAL_DIR}/dlibx/dnn_loss_metric.cpp
        ${LIB_INTERNAL_DIR}/dlibx/dnn_loss_cosine.cpp
        ${LIB_INTERNAL_DIR}/dlibx/blas_openblas.cpp
        ${LIB_INTERNAL_DIR}/dlibx/matrix_ops_generic.cpp
)
set_target_properties(dlibx PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(dlibx PRIVATE core json raw_image dlib openblas)
target_include_directories(dlibx PRIVATE ${LIB_INTERNAL_DIR})

############
# DET_DLIB #
############

add_library(det_dlib STATIC
        ${LIB_INTERNAL_DIR}/det_dlib/classifiers.cpp
        ${LIB_INTERNAL_DIR}/det_dlib/types.cpp
        ${LIB_INTERNAL_DIR}/det_dlib/init.cpp
)
set_target_properties(det_dlib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(det_dlib PRIVATE det dlibx openblas)
target_include_directories(det_dlib PRIVATE ${LIB_INTERNAL_DIR})

# ============ Android Shared Libraries =============

add_library(FaceClassification SHARED
        FaceClassifier.cpp
        conversion.cpp
        jniglue.cpp
)
target_include_directories(FaceClassification PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${LIB_INTERNAL_DIR})
target_link_libraries(FaceClassification PRIVATE det_dlib)

set_target_properties(FaceClassification PROPERTIES PREFIX "lib" SUFFIX ".so")