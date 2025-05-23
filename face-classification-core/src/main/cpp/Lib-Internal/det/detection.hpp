#pragma once

#include "types.hpp"
#include "detection_internal.hpp"
#include <models/loader.hpp>
#include <stdext/arg.hpp>
#include <stdext/options_tuple.hpp>

namespace det {

    /** \name Initialization.
     */
    //@{
    /** \brief Set a models loader method for detection.
     *
     * This method must be called before doing any detection.
     * Once set for the specified context, the loader cannot be changed.
     */
    void set_models_loader(stdx::arg<core::context> context,
                           models::loader_function models_loader);

    /** \brief Set models loader to load from a models directory.
     *
     * The PATH object may be either std::filesystem::path or
     * boost::filesystem::path, provided support has been compiled into
     * the models library.
     */
    template <typename PATH>
    inline std::enable_if_t<stdx::is_path_v<PATH> >
    set_models_path(stdx::arg<core::context> context, PATH models_path) {
        set_models_loader(
            context, models::loader<PATH>(std::move(models_path)));
    }

    /** \brief Prepare context for face detection.
     *
     * Start face detection threads and load necessary data.
     *
     * The use of this method is optional as threads will be started
     * automatically and necessary data will be loaded as needed.
     * If there is any problem loading data, this method will throw
     * an exception.
     */
    void prepare_detection(stdx::arg<core::context> context,
                           const detection_settings& settings);
    inline void prepare_detection(stdx::arg<core::context> context,
                                  const detection_settings& settings,
                                  models::loader_function models_loader) {
        set_models_loader(context, std::move(models_loader));
        prepare_detection(context, settings);
    }
    template <typename PATH>
    inline std::enable_if_t<stdx::is_path_v<PATH> >
    prepare_detection(stdx::arg<core::context> context,
                      const detection_settings& settings,
                      PATH models_path) {
        set_models_path(context, std::move(models_path));
        prepare_detection(context, settings);
    }
    //@}
    
    
    /** \name Detection.
     */
    //@{

    /** \brief Handle object for asynchronous detection.
     */
    template <typename RESULT>
    class detection_handle {
        internal::detection_state_ptr handle;

    public:
        detection_handle() = default;
        detection_handle(internal::detection_state_ptr&& handle)
            : handle(move(handle)) {}
        detection_handle(detection_handle&&) = default;
        detection_handle& operator=(detection_handle&&) = default;

        explicit operator bool() const { return handle.get(); }
        
        std::vector<RESULT> get_some() {
            auto vec = internal::get_some(*handle);
            std::vector<RESULT> result;
            result.reserve(vec.size());
            for (auto& a : vec)
                result.emplace_back(std::move(*static_cast<RESULT*>(a.get())));
            return result;
        }
        std::vector<RESULT> get_all() {
            std::vector<RESULT> result;
            for (;;) {
                auto vec = internal::get_some(*handle);
                if (vec.empty()) break;
                for (auto& a : vec)
                    result.emplace_back(
                        std::move(*static_cast<RESULT*>(a.get())));
            }
            return result;
        }

        class iterator {
            detection_handle* h;
            std::vector<RESULT> vec;
            typename std::vector<RESULT>::iterator iter;

        public:
            using value_type = RESULT;
            using reference = RESULT&;
            using pointer = RESULT*;
            using difference_type = std::ptrdiff_t;
            using iterator_category = std::input_iterator_tag;

            iterator(detection_handle* h = nullptr)
                : h(h),
                  vec(h ? h->get_some() : std::vector<RESULT>{}),
                  iter(vec.begin()) {
                if (vec.empty()) this->h = nullptr;
            }
            iterator(iterator&& other)
                : h(other.h),
                  vec(move(other.vec)),
                  iter(other.iter) {
                other.h = nullptr;
            }
            iterator(const iterator& other)
                : h(other.h),
                  vec(other.vec),
                  iter(vec.begin() + (other.iter - other.vec.begin())) {
            }
            iterator& operator=(iterator&& other) {
                if (this != &other) {
                    h = other.h;
                    vec = move(other.vec);
                    iter = other.iter;
                    other.h = nullptr;
                }
                return *this;
            }
            iterator& operator=(const iterator& other) {
                if (this != &other) {
                    h = other.h;
                    vec = other.vec;
                    iter = vec.begin() + (other.iter - other.vec.begin());
                }
                return *this;
            }

            inline bool operator==(const iterator& other) {
                return h ? this == &other : !other.h;
            }
            inline bool operator!=(const iterator& other) {
                return !(*this == other);
            }

            iterator& operator++() {
                if (++iter == vec.end()) {
                    vec = h->get_some();
                    iter = vec.begin();
                    if (vec.empty())
                        h = nullptr;
                }
                return *this;
            }
            iterator operator++(int) {
                auto prev = *this;
                ++*this;
                return prev;
            }

            inline reference operator*() const { return *iter; }
            inline pointer operator->() const { return &*iter; }
        };

        auto begin() { return iterator{this}; }
        auto end() { return iterator{}; }
    };

    /** \brief Low latency option.
     *
     * If low_latency is selected, then jobs are scheduled so as to return
     * the first face found as soon as possible.
     * If the batch option is selected, then jobs are scheduled
     * for maximum throughput (best use of processor cores).
     */
    struct low_latency_tag;
    using low_latency_option = stdx::option_bool<low_latency_tag>;
    const low_latency_option low_latency{true};
    const low_latency_option batch{false};

    /** \brief Asynchronous complete face detection with output constructor.
     *
     * The output constructor object must have the following form:
     *
     * struct example_output_constructor {
     *     example_output_constructor(const example_output_constructor&,
     *                                const face_coordinates&,
     *                                core::job_context&);
     *     OUTPUT operator()(face_coordinates&, core::job_context&);
     * };
     *
     * The copy constructor is called for each detected face with the
     * coordinates from face detection.
     * Then, after landmark detection, and from the same thread, the
     * output operator is called to complete construction of the output.
     * The face_coordinates vector passed to the output operator contains
     * all detected and landmark coordinates.
     */
    template <typename OUTPUT_CONSTRUCTOR>
    detection_handle<internal::output_type<OUTPUT_CONSTRUCTOR> >
    start_detect_faces(
        core::active_job context,
        const detection_settings& settings,
        stdx::arg<const image_struct> image,
        OUTPUT_CONSTRUCTOR&& output_constructor,
        low_latency_option latency_option = batch,
        json::value* diagnostic = nullptr) {

        using outfn = internal::output_fn<std::decay_t<OUTPUT_CONSTRUCTOR> >;
        return internal::start_detect_faces(
            context, settings, image.get(),
            std::make_unique<outfn>(
                std::forward<OUTPUT_CONSTRUCTOR>(output_constructor)),
            latency_option, diagnostic);
    }

    /** \brief Asynchronous complete face detection.
     *
     * This method will return immediately and if num_threads > 1,
     * detection will begin in the background.
     *
     * The returned handle is to be used with get_some_faces().
     */
    detection_handle<face_coordinates>
    start_detect_faces(
        core::active_job context,
        const detection_settings& settings,
        stdx::arg<const image_struct> image,
        low_latency_option latency_option = low_latency,
        json::value* diagnostic = nullptr);

    /** \brief Detect faces.
     *
     * \see start_detect_faces() for details
     */
    inline auto detect_faces(
        core::active_job context,
        const detection_settings& settings,
        stdx::arg<const image_struct> image,
        json::value* diagnostic = nullptr) {
        return start_detect_faces(
            std::move(context), settings, image,
            batch, diagnostic).get_all();
    }
    template <typename OUTPUT_CONSTRUCTOR>
    inline auto detect_faces(
        core::active_job context,
        const detection_settings& settings,
        stdx::arg<const image_struct> image,
        OUTPUT_CONSTRUCTOR&& output_constructor,
        json::value* diagnostic = nullptr) {
        return start_detect_faces(
            std::move(context), settings, image,
            std::forward<OUTPUT_CONSTRUCTOR>(output_constructor),
            batch,diagnostic).get_all();
    }


    /** \brief Asynchronous landmark detection with output constructor.
     *
     * \see start_detect_faces() for description of output constructor
     * Note that the copy constructor of the output constructor will
     * receive the initial coordinates passed to this method.
     */
    template <typename OUTPUT_CONSTRUCTOR, typename ITER>
    detection_handle<internal::output_type<OUTPUT_CONSTRUCTOR> >
    start_detect_landmarks(
        core::active_job context,
        const landmark_settings& landmarks,
        stdx::arg<const image_struct> image,
        ITER first, ITER last,
        OUTPUT_CONSTRUCTOR&& output_constructor) {

        using outfn = internal::output_fn<std::decay_t<OUTPUT_CONSTRUCTOR> >;
        return internal::start_detect_landmarks(
            context, landmarks, image.get(),
            std::move(first), std::move(last),
            std::make_unique<outfn>(
                std::forward<OUTPUT_CONSTRUCTOR>(output_constructor)));
    }

    /** \brief Asynchronous landmark detection.
     *
     * This method will return immediately and if num_threads > 1,
     * detection will begin in the background.
     *
     * The returned handle is to be used with get_some_faces().
     *
     * This method takes a list of face coordinate estimates and performs
     * the desired landmark detection at each location.
     * The number of faces returned will equal the number estimates provided
     * and they will be returned in the same order.
     * Note though that get_some_faces() may return the faces a few at a time.
     */
    detection_handle<face_coordinates>
    start_detect_landmarks(
        core::active_job context,
        const landmark_settings& landmarks,
        stdx::arg<const image_struct> image,
        stdx::forward_iterator<const detected_coordinates&> first, 
        stdx::forward_iterator<const detected_coordinates&> last);

    /** \brief Detect landmarks.
     *
     * \see start_detect_landmarks() for details
     */
    face_list_type detect_landmarks(
        core::active_job context,
        const landmark_settings& landmarks,
        stdx::arg<const image_struct> image,
        stdx::forward_iterator<const detected_coordinates&> first, 
        stdx::forward_iterator<const detected_coordinates&> last);

    
    /** \brief Get detected faces.
     *
     * This method is to be called repeatedly to retrieve the results of
     * face detection begun by start_detect_faces().
     * On each call, the method will block until at least one face is ready
     * to be returned.
     * If an empty list is returned, then face detection is complete.
     * More than one face may be returned per call.
     *
     * Note that this method does not "block" in the sense that it
     * stalls or waits.  
     * Instead, it participates in the work that needs to be done.
     * If num_threads == 1, then this method does all the work.
     */
    template <typename T>
    std::vector<T> get_some_faces(detection_handle<T>& handle) {
        return handle.get_some();
    }
    template <typename T>
    std::vector<T> get_some(detection_handle<T>& handle) {
        return handle.get_some();
    }
    /// get all detected faces
    template <typename T>
    std::vector<T> get_all(detection_handle<T>& handle) {
        return handle.get_all();
    }
    //@}
    
}
