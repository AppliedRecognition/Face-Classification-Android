#pragma once

//
// Created by Jakub Dolejs on 08/01/2019.
//

#include <raw_image/types.hpp>
#include <det/types.hpp>
#include <det_dlib/types.hpp>
#include <rec/types.hpp>
#include <render/types.hpp>
#include <stdext/span.hpp>


struct FaceSize {
    float width, height;
};

/** \brief Raw image from pixel buffer.
 *
 * The pixels are accessed in place (not copied) so the caller must
 * ensure the pixel buffer remains valid for the life of this object.
 *
 * \param buffer Image buffer
 * \param width Width of the image in pixels
 * \param height Height of the image in pixels
 * \param layout Image colour space
 * \param bytes_per_row Bytes per row (default 0 will set the value to the image width * bytes per pixel)
 * \return pointer to raw image structure
 */
std::unique_ptr<stdx::span<raw_image::plane> >
rawImageFromBuffer(void* buffer, unsigned width, unsigned height,
                   raw_image::pixel_layout layout,
                   unsigned bytes_per_row = 0);

/** \brief Grayscale image from raw pixel buffer.
 *
 * This is just a special case of rawImageFromBuffer().
 */
inline auto
grayscaleImageFromBuffer(void* buffer, unsigned width, unsigned height,
                         unsigned bytes_per_row = 0) {
    return rawImageFromBuffer(
        buffer, width, height, raw_image::pixel::gray8, bytes_per_row);
}

/** \brief Android NV21 multi-plane image from buffer.
 *
 * The buffer is expected to be packed (no padding) with 2 images:
 *   width x height 8-bit per pixel Y image followed by
 *   width/2 x height/2 16-bit per pixel VU image.
 *
 * The pixels are accessed in place (not copied) so the caller must
 * ensure the pixel buffer remains valid for the life of this object.
 */
std::unique_ptr<stdx::span<raw_image::plane> >
NV21ImageFromBuffer(void* buffer, unsigned width, unsigned height, unsigned bytes_per_row);




/** \brief Faces being tracked through successive frames.
 *
 * To start a session, default construct an object of this type and
 * use it for every call to trackFaces().
 */
using FaceTrackingSession = std::vector<det::detected_coordinates>;


/** \brief List of faces: coordinates and classifiers.
 */
using face_list_type = std::vector<det::face_coordinates_with_classifiers>;


/** \brief Context and settings necessary for face detection and recognition.
 */
class FaceDetectionRecognition {
public:
    std::unique_ptr<core::context> context;
    det::detection_settings settings;
    std::vector<det::classifier_model_type const*> detection_classifiers;
    float faceExtractQualityThreshold;
    float landmarkTrackingQualityThreshold;


    /** \brief Constructor.
     */
    FaceDetectionRecognition(
        std::string modelsPath,
        det::detection_settings settings,
        float faceExtractQualityThreshold = 8.0f,
        float landmarkTrackingQualityThreshold = 8.0f);


    /// destructor and move
    ~FaceDetectionRecognition();
    FaceDetectionRecognition(FaceDetectionRecognition&&);
    FaceDetectionRecognition& operator=(FaceDetectionRecognition&&);


    /** \name Face Detection
     */
    //@{

    /** \brief Find and load model file for classifier.
     *
     * On failure an exception is thrown.
     */
    det::classifier_model_type const*
    loadClassifier(const std::string& name);
    
    det::classifier_model_type const*
    loadClassifier(const std::string &name, const stdx::binary &data);

    /** \brief Find and load model file for classifier.
     *
     * On success the classifier is added to the detection_classifiers vector.
     * On failure an exception is thrown.
     */
    inline void loadDetectionClassifier(const std::string& name) {
        detection_classifiers.push_back(loadClassifier(name));
    }
    
    void loadModelFile(const int faceTemplateVersion);

    /** \brief Detect all faces in image.
     *
     * All faces found meeting landmarkTrackingQualityThreshold
     * will be returned.
     * The effect of this method is the same as trackFaces() with
     * a new or empty session.
     */
    face_list_type detectFaces(
        const stdx::spanarg<const raw_image::plane>& image);

    /** \brief Track faces using face detection and landmark detection.
     *
     * This method will do full face detection if either
     * no faces are currently being tracked, or
     * the tracking on at least one face has been lost.
     * Otherwise landmark detection is used to track the faces.
     *
     * The tracking is considered lost if the quality falls below
     * landmarkTrackingQualityThreshold.
     */
    face_list_type trackFaces(
        const stdx::spanarg<const raw_image::plane>& image,
        FaceTrackingSession& session);

    /** \brief Find face with largest product of size and quality.
     *
     * The product of face size (distance between the eyes) and quality
     * is used as the measure of "dominance".
     *
     * If at least one face meets faceExtractQualityThreshold, then
     * such a face will be returned.
     * Otherwise, a face not meeting the threshold will be returned.
     *
     * \throws std::invalid_argument if faces is empty
     */
    const det::face_coordinates_with_classifiers&
    findDominantFace(const face_list_type& faces) const;
    //@}


    /** \name Face Classifiers
     */
    //@{

    static float qualityFromFace(const det::face_coordinates& fc);

    static float eyeDistanceFromFace(const det::face_coordinates& fc);
    static inline auto dimensionsOfFace(const det::face_coordinates& fc) {
        const auto eyeDistance = eyeDistanceFromFace(fc);
        return FaceSize { 2.96f * eyeDistance, 3.70f * eyeDistance };
    }

    static det::coordinate_type centerOfFace(const det::face_coordinates& fc);

    /** \brief Apply classifier model to face and return result.
     */
    std::vector<float> extractClassifier(
        const stdx::spanarg<const raw_image::plane>& image,
        const det::face_coordinates& fc,
        det::classifier_model_type const* ap) const;

    /** \brief Align 3d average face model to detected landmarks.
     *
     * This method computes rotation parameters along with position of
     * face in 3-d space assuming focal_length is reasonably accurate.
     *
     * The version that accepts the image as input only uses the image
     * to get width and height.  
     * Ensure the rotate value is set correctly so width and height are
     * not swapped.
     *
     * If focal_length == 0, then it defaults to max(width,height).
     */
    render::face_alignment facePoseFromLandmarks(
        const det::face_coordinates& face,
        unsigned width, unsigned height,
        unsigned focal_length = 0);
    render::face_alignment facePoseFromLandmarks(
        const det::face_coordinates& face,
        const stdx::spanarg<const raw_image::plane>& image,
        unsigned focal_length = 0);
    //@}


    /** \name Template Extraction
     */
    //@{

    /** \brief Extract recognition template from image.
     *
     * The template is only extracted if the face meets the
     * faceExtractQualityThreshold.
     * Otherwise, nullptr is returned.
     */
    rec::prototype_ptr extractTemplate(
        const stdx::spanarg<const raw_image::plane>& image,
        const det::face_coordinates& face,
        unsigned templateVersion);

    /** \brief Serialize template to binary data.
     */
    static stdx::binary serializeTemplate(const rec::prototype_ptr& p);
    stdx::binary serializeRawTemplate(const rec::prototype_ptr& p) const;
    //@}


    /** \name Template Comparison
     */
    //@{

    /** \brief Deserialize template.
     */
    rec::prototype_ptr createFaceFromData(const stdx::binary& data) const;

    /** \brief Deserialize subject.
     *
     * If given a single serialized prototype as input, this method will
     * deserialize the template and create a subject with that one
     * template in it.
     */
    rec::multiface_ptr createSubjectFromData(const stdx::binary& data) const;
    
    /** \brief Create subject from list of faces.
     */
    static rec::multiface_ptr createSubjectFromFaces(
        const std::vector<rec::prototype_ptr>& faces);

    /** \brief Compare subject to multiple faces.
     *
     * The maximum of the scores is returned.
     * If faces is empty, 0 is returned.
     */
    static float compareSubjectToFaces(
        const rec::multiface_ptr& subject,
        const std::vector<rec::prototype_ptr>& faces);
    //@}
};
