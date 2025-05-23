//
// Created by Jakub Dolejs on 08/01/2019.
//

#include "FaceDetectionRecognition.hpp"

#include <det_ncnn/init.hpp>
#include <det_dlib/init.hpp>
#include <det/image.hpp>
#include <det/classifiers.hpp>
#include <det/landmark_standardize.hpp>
#include <rec/prototype.hpp>
#include <rec/multiface.hpp>
#include <rec_ncnn/engine.hpp>
#include <rec_dlib/engine.hpp>
#include <render/frontalize.hpp>
#include <core/context.hpp>
#include <raw_image/transform.hpp>
#include <raw_image/point_rounding.hpp>
#include <filesystem>

static auto contextSettings() {
    core::context_settings cs;
    cs.max_threads = 8;  // auto detect
    return cs;
}

std::unique_ptr<stdx::span<raw_image::plane> >
rawImageFromBuffer(void* buffer, unsigned width, unsigned height,
                   raw_image::pixel_layout layout,
                   unsigned bytes_per_row) {
    if (!buffer || width <= 0 || height <= 0)
        throw std::invalid_argument("image buffer is empty or null pointer");
    const auto bytesPerPixel = bytes_per_pixel(layout);
    if (bytesPerPixel <= 0 || 4 < bytesPerPixel)
        throw std::invalid_argument("invalid pixel layout");
    struct s {
        stdx::span<raw_image::plane> s;
        raw_image::plane r;
    };
    auto p = new s;
    p->s = { &p->r, 1 };
    p->r.width = width;
    p->r.height = height;
    p->r.layout = layout;
    p->r.bytes_per_line = std::max(width*bytesPerPixel,bytes_per_row);
    p->r.data = static_cast<unsigned char*>(buffer);
    assert(static_cast<void*>(p) == static_cast<void*>(&p->s));
    return std::unique_ptr<stdx::span<raw_image::plane> >(&p->s);
}

std::unique_ptr<stdx::span<raw_image::plane> >
NV21ImageFromBuffer(void* buffer, unsigned width, unsigned height, unsigned bytes_per_row) {
    if (!buffer || width <= 0 || height <= 0)
        throw std::invalid_argument("image buffer is empty or null pointer");
    struct s {
        stdx::span<raw_image::plane> s;
        std::array<raw_image::plane,2> a;
    };
    auto p = new s;
    p->s = p->a;
    auto& a = p->a;
    a[0].data = static_cast<unsigned char*>(buffer);
    a[0].width = width;
    a[0].height = height;
    a[0].bytes_per_line = bytes_per_row;
    a[0].layout = raw_image::pixel::y8_nv21;
    a[1].data = a[0].data + a[0].height*a[0].bytes_per_line;
    a[1].width = a[0].width/2;
    a[1].height = a[0].height/2;
    a[1].bytes_per_line = bytes_per_row;
    a[1].layout = raw_image::pixel::vu16_nv21;
    assert(static_cast<void*>(p) == static_cast<void*>(&p->s));
    return std::unique_ptr<stdx::span<raw_image::plane> >(&p->s);
}

FaceDetectionRecognition::FaceDetectionRecognition(
    std::string modelsPath, det::detection_settings settings,
    float faceExtractQualityThreshold,
    float landmarkTrackingQualityThreshold)
    : context(core::context::construct(contextSettings())),
      settings(settings),
      faceExtractQualityThreshold(faceExtractQualityThreshold),
      landmarkTrackingQualityThreshold(landmarkTrackingQualityThreshold) {
    det::dlib::init(context);
    det::ncnn::init(context);
    const auto mp = std::filesystem::path(modelsPath);
    det::prepare_detection(context, settings, mp);
    rec::ncnn::initialize(context, mp);
    rec::dlib::initialize(context, mp);
}

FaceDetectionRecognition::~FaceDetectionRecognition() = default;
FaceDetectionRecognition::FaceDetectionRecognition(FaceDetectionRecognition&&) = default;
FaceDetectionRecognition& FaceDetectionRecognition::operator=(FaceDetectionRecognition&&) = default;

float
FaceDetectionRecognition::qualityFromFace(const det::face_coordinates& fc) {
    if (fc.empty() || fc.back().landmarks.size() <= 2)
        return 0.0f;
    return fc.back().confidence;
}

float
FaceDetectionRecognition::eyeDistanceFromFace(const det::face_coordinates& fc) {
    if (fc.empty()) return 0.0f;
    auto& dc = fc.back();
    const auto dx = dc.eye_left.x - dc.eye_right.x;
    const auto dy = dc.eye_left.y - dc.eye_right.y;
    return std::sqrt(dx*dx + dy*dy);
}

det::coordinate_type FaceDetectionRecognition::centerOfFace(
    const det::face_coordinates& fc) {
    if (fc.empty()) return {0,0};
    auto& dc = fc.back();
    const auto ls = det::landmark_standardize<>(dc.eye_left, dc.eye_right);
    auto center = ls.eye_center;
    if (ls.eye_dist >= 1)
        center += 0.26f*ls.eye_dist*ls.down;
    return center;
}

det::classifier_model_type const*
FaceDetectionRecognition::loadClassifier(const std::string& name) {
    if (auto p = det::load_classifier(context, name))
        return p;
    else throw std::runtime_error(
        "det::load_classifier() returned nullptr");
}

det::classifier_model_type const*
FaceDetectionRecognition::loadClassifier(const std::string &name, const stdx::binary &data) {
    if (auto p = det::load_classifier(context, name, data, name))
        return p;
    else throw std::runtime_error(
                                  "det::load_classifier() returned nullptr");
}

std::vector<float> FaceDetectionRecognition::extractClassifier(
    const stdx::spanarg<const raw_image::plane>& image,
    const det::face_coordinates& face,
    det::classifier_model_type const* ap) const {
    return apply_classifier(context, ap, image, face);
}

void FaceDetectionRecognition::loadModelFile(const int faceTemplateVersion) {
    rec::prototype::load_model(context, faceTemplateVersion);
}

face_list_type FaceDetectionRecognition::detectFaces(
    const stdx::spanarg<const raw_image::plane>& raw) {

    if (raw.empty())
        throw std::invalid_argument("multi-plane image is empty");

    det::apply_classifiers da(raw, detection_classifiers, {});

    const auto image = det::share_pixels(context, settings, raw);

    face_list_type detected;
    auto h = start_detect_faces(
        context, settings, image, std::move(da), det::batch);
    std::move(h.begin(), h.end(), back_inserter(detected));

    // remove faces that don't meet the quality threshold
    const auto end = remove_if(
        detected.begin(), detected.end(),
        [&](const auto& face) {
            return qualityFromFace(face) < landmarkTrackingQualityThreshold;
        });
    detected.erase(end, detected.end());

    return detected;
}

face_list_type FaceDetectionRecognition::trackFaces(
    const stdx::spanarg<const raw_image::plane>& raw,
    FaceTrackingSession& candidates) {

    if (candidates.empty()) {
        auto faces = detectFaces(raw);
        for (auto& face : faces)
            candidates.emplace_back(face);
        return faces;
    }

    auto s = settings;
    s.detector_version = 0;
    const auto image = det::share_pixels(context, s, raw);

    face_list_type faces;
    faces.reserve(candidates.size());
    auto h = start_detect_landmarks(
        context, settings.landmark_detection, image,
        candidates.begin(), candidates.end());
    for (auto&& face : h)
        faces.emplace_back(std::move(face));

    candidates.clear();
    for (auto& face : faces) {
        // save as previous for the next frame,
        // but only if face is of sufficient quality
        if (qualityFromFace(face) >= landmarkTrackingQualityThreshold)
            candidates.push_back(face);
    }
    if (candidates.size() < faces.size())
        candidates.clear();  // trigger full detection next frame
    return faces;
}

const det::face_coordinates_with_classifiers&
FaceDetectionRecognition::findDominantFace(const face_list_type& faces) const {
    if (faces.empty())
        throw std::invalid_argument("empty faces vector");
    return *max_element(
        faces.begin(), faces.end(),
        [qt=faceExtractQualityThreshold](const auto& a, const auto& b) {
            auto qa = qualityFromFace(a);
            auto qb = qualityFromFace(b);
            if ((qa < qt && qb < qt) || (qa >= qt && qb >= qt)) {
                qa *= eyeDistanceFromFace(a);
                qb *= eyeDistanceFromFace(b);
            }
            return qa < qb;
        });
}

render::face_alignment FaceDetectionRecognition::facePoseFromLandmarks(
    const det::face_coordinates& face,
    unsigned width, unsigned height,
    unsigned focal_length) {
    return render::align_model(context, face, {width,height}, focal_length);
}
render::face_alignment FaceDetectionRecognition::facePoseFromLandmarks(
    const det::face_coordinates& face,
    const stdx::spanarg<const raw_image::plane>& image,
    unsigned focal_length) {
    const auto dims = dimensions(image);
    return render::align_model(context, face, dims, focal_length);
}

rec::prototype_ptr FaceDetectionRecognition::extractTemplate(
    const stdx::spanarg<const raw_image::plane>& image,
    const det::face_coordinates& face,
    unsigned templateVersion) {
    if (qualityFromFace(face) >= faceExtractQualityThreshold)
        return rec::prototype::extract(context, image, face, templateVersion);
    return nullptr;
}

stdx::binary FaceDetectionRecognition::serializeTemplate(
    const rec::prototype_ptr& p) {
    return to_binary(p);
}
stdx::binary FaceDetectionRecognition::serializeRawTemplate(
    const rec::prototype_ptr& p) const {
    rec::prototype::set_serialize_format(context, p->version, 1);
    return to_binary(p, rec::raw);
}

rec::prototype_ptr FaceDetectionRecognition::createFaceFromData(
    const stdx::binary& data) const {
    return rec::prototype::deserialize(context, data);
}

rec::multiface_ptr FaceDetectionRecognition::createSubjectFromData(
    const stdx::binary& data) const {
    return rec::multiface(context, data).release();
}

rec::multiface_ptr FaceDetectionRecognition::createSubjectFromFaces(
    const std::vector<rec::prototype_ptr>& faces) {
    return rec::multiface(faces.begin(), faces.end()).release();
}

float FaceDetectionRecognition::compareSubjectToFaces(
    const rec::multiface_ptr& mf,
    const std::vector<rec::prototype_ptr>& faces) {
    float score = 0;
    for (auto& face : faces) {
        float result = compare(mf, face);
        if (result > score)
            score = result;
    }
    return score;
}

