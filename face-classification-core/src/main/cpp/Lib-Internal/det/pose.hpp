#pragma once

#include "types.hpp"

namespace det {

    /** \brief Yaw and pitch calculation methods.
     *
     * The nose_tip method is very simple.  It estimates yaw and pitch
     * from the x,y location of the tip of the nose relative to the eyes.
     *
     * The simplex method uses up to 25 iterations of a simplex method
     * to fit both the tip and base of the nose to a general 3-d model
     * of the human face.
     */
    enum class pose_method { nose_tip = 1, simplex = 2 };


    /** \brief Compute yaw, pitch and roll from detected coordinates.
     *
     * This method requires dlib68 landmarks and these landmarks are
     * relative to the image.
     */
    template <pose_method method = pose_method::simplex>
    face_pose_type compute_pose(const face_coordinates& face);


    /** \brief Compute yaw and pitch from standardized landmarks.
     *
     * Requires dlib68 landmarks standardized using landmark_standarize.
     * \sa landmark_standardize.hpp for definition of standardized landmarks
     */
    template <pose_method method = pose_method::simplex>
    face_pose_type compute_pose(
        const std::vector<coordinate_type>& standardized_landmarks);


    /** \brief Select base of nose or center of mouth landmark.
     */
    enum class base_landmark_type { nose = 1, mouth = 2 };
    constexpr auto nose = base_landmark_type::nose;
    constexpr auto mouth = base_landmark_type::mouth;


    /** \brief Compute yaw, pitch and roll from 4 landmarks.
     *
     * The 4 landmarks are relative to the image.
     * The eye locations are center of eye, which is
     * the midpoint between the corners.
     *
     * The base coordinate can be either base of the nose or center of
     * the mouth.  Center of the mouth is the midpoint between the corners.
     *
     * Note that the nose_tip method does not use the base landmark.
     */
    template <pose_method method = pose_method::simplex>
    face_pose_type compute_pose(
        const coordinate_type& eye_left, const coordinate_type& eye_right,
        const coordinate_type& nose_tip,
        const coordinate_type& base, base_landmark_type type);


    /** \brief Compute yaw and pitch from standardize landmarks.
     *
     * The 2 landmarks specified must be standardized using landmark_standarize.
     * \sa landmark_standardize.hpp for definition of standardized landmarks
     */
    template <pose_method method = pose_method::simplex>
    face_pose_type compute_pose(
        const coordinate_type& nose_tip,
        const coordinate_type& base, base_landmark_type type);
}
