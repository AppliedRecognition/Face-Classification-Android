#include <boost/test/unit_test.hpp>
#include <applog/base_directory.hpp>
#include <applog/core.hpp>

#include <core/context.hpp>
#include <raw_image_io/io.hpp>
#include <raw_image/transform.hpp>

#include <det/image.hpp>
#define __FBLIB_DET_PRIVATE_INTERNAL_USE_ONLY__
#include <det/internal_image.hpp>

using namespace det;

void* operator new(std::size_t sz) {
    if (sz == 0)
        ++sz; // avoid std::malloc(0) which may return nullptr on success
    if (void *ptr = std::malloc(sz))
        return ptr;
    throw std::bad_alloc{}; // required by [new.delete.single]/3
}

static void* s_ptr_watched;
static unsigned s_ptr_deleted;

#pragma GCC diagnostic ignored "-Wsized-deallocation"
void operator delete(void* ptr) noexcept {
    if (ptr == s_ptr_watched) {
        ++s_ptr_deleted;
        //std::puts("global op delete called");
    }
    std::free(ptr);
}

BOOST_AUTO_TEST_SUITE(det)

static bool raw_image_same_meta(const raw_image::plane& a,
                                const raw_image::plane& b) {
    throw_if_invalid(a);
    throw_if_invalid(b);
    return a.width == b.width && a.height == b.height &&
        a.layout == b.layout &&
        a.rotate == b.rotate && a.scale == b.scale;
}

static bool raw_image_same_pixels(const raw_image::plane& a,
                                  const raw_image::plane& b) {
    if (!raw_image_same_meta(a,b)) return false;
    const auto n = a.width * bytes_per_pixel(a.layout);
    for (unsigned y = 0; y < a.height; ++y) {
        auto ap = a.data + y*a.bytes_per_line;
        auto bp = b.data + y*b.bytes_per_line;
        if (memcmp(ap, bp, n) != 0) return false;
    }
    return true;
}

static inline bool is_gray(const raw_image::plane& img) {
    return same_channel_order(img.layout, raw_image::pixel::gray8);
}
static inline bool is_color(const raw_image::plane& img) {
    return bytes_per_pixel(img.layout) > 1;
}
static inline bool is_rotated(const raw_image::plane& img) {
    return img.rotate & 3;
}

static void _color(const image_type& image) {
    BOOST_CHECK(is_color(get_raw_from_image(image, gray)));
    BOOST_CHECK(is_color(get_raw_from_image(image, color)));
}
static void _gray(const image_type& image) {
    BOOST_CHECK(is_gray(get_raw_from_image(image, gray)));
    BOOST_CHECK(is_gray(get_raw_from_image(image, color)));
}
static void _both(const image_type& image) {
    BOOST_CHECK(is_gray(get_raw_from_image(image, gray)));
    BOOST_CHECK(is_color(get_raw_from_image(image, color)));
}

static void _rot(const image_type& image) {
    BOOST_CHECK(is_rotated(get_raw_from_image(image, gray)));
    BOOST_CHECK(is_rotated(get_raw_from_image(image, color)));
}
static void _norot(const image_type& image) {
    BOOST_CHECK(!is_rotated(get_raw_from_image(image, gray)));
    BOOST_CHECK(!is_rotated(get_raw_from_image(image, color)));
}
static void _norotg(const image_type& image) {
    BOOST_CHECK(!is_rotated(get_raw_from_image(image, gray)));
}
static void _norotc(const image_type& image) {
    BOOST_CHECK(!is_rotated(get_raw_from_image(image, color)));
}

template <typename T>
static inline auto watchptr(std::unique_ptr<T> ptr) {
    s_ptr_watched = ptr.get();
    s_ptr_deleted = 0;
    return ptr;
}
static void _ptrgood(const image_type& image) {
    BOOST_CHECK(s_ptr_watched);
    BOOST_CHECK_EQUAL(s_ptr_deleted, 0);
    auto data = static_cast<const raw_image::plane*>(s_ptr_watched)->data;
    BOOST_CHECK(get_raw_from_image(image, gray).data == data ||
                get_raw_from_image(image,color).data == data);
}

static const image_type& operator>>(const image_type& image,
                                    void(&check)(const image_type&)) {
    check(image);
    return image;
}

struct _share { stdx::arg<const raw_image::plane> raw; };
static const image_type& operator>>(const image_type& image, _share s) {
    BOOST_CHECK(get_raw_from_image(image,gray).data  == s.raw->data ||
                get_raw_from_image(image,color).data == s.raw->data);
    return image;
}
struct _noshare { stdx::arg<const raw_image::plane> raw; };
static const image_type& operator>>(const image_type& image, _noshare s) {
    BOOST_CHECK(get_raw_from_image(image,gray).data  != s.raw->data &&
                get_raw_from_image(image,color).data != s.raw->data);
    return image;
}

struct _ptrshared { std::shared_ptr<const raw_image::plane> const& ptr; };
static const image_type& operator>>(const image_type& image, _ptrshared s) {
    BOOST_CHECK_EQUAL(s.ptr.use_count(), 2);
    return image;
}


BOOST_AUTO_TEST_CASE(det_image) {
    const auto base_path = base_directory("lib-internal") / "det" / "tests";
    const auto img_path = base_path / "image_077.jpg";
    
    FILE_LOG(logINFO) << "image: start";
    core::context_settings cs;
    auto c = core::context::construct(cs);

    const auto raw = raw_image::load(img_path);
    BOOST_REQUIRE(is_color(*raw) && !is_rotated(*raw));
    const auto sraw =
        std::shared_ptr<const raw_image::plane>(raw.get(),[](auto){});
    const auto raw_orig = copy(raw);

    const auto rot = copy_rotate(*raw,1);
    BOOST_REQUIRE(is_rotated(*rot));
    const auto srot =
        std::shared_ptr<const raw_image::plane>(rot.get(),[](auto){});
    const auto rot_orig = copy(rot);

    {
        detection_settings s;
        s.detector_version = 0;

        /*
        FILE_LOG(logDETAIL) << "load";
        load_image(c,s,img_path,rotate(1)) >> _gray >> _norot;
        load_image(c,s,img_path,gray) >> _gray >> _norot;
        load_image(c,s,img_path,color) >> _color >> _norot;
        load_image(c,s,img_path,color,gray) >> _both >> _norot;
        */

        // copy image
        FILE_LOG(logDETAIL) << "copy_image";
        copy_image(c,s,raw) >> _gray >> _norot >> _noshare{raw};
        copy_image(c,s,raw,gray) >> _gray >> _norot >> _noshare{raw};
        copy_image(c,s,raw,color) >> _color >> _norot >> _noshare{raw};
        copy_image(c,s,raw,color,gray) >> _both >> _norot >> _noshare{raw};

        copy_image(c,s,rot) >> _gray >> _rot >> _noshare{rot};
        copy_image(c,s,rot,gray) >> _gray >> _rot >> _noshare{rot};
        copy_image(c,s,rot,color) >> _color >> _rot >> _noshare{rot};
        copy_image(c,s,rot,color,gray) >> _both >> _rot >> _noshare{rot};

        // use pixels (no modify)
        FILE_LOG(logDETAIL) << "share_pixels";
        share_pixels(c,s,raw) >> _color >> _norot >> _share{raw};
        share_pixels(c,s,raw,gray) >> _gray >> _norot >> _noshare{raw};
        share_pixels(c,s,raw,color) >> _color >> _norot >> _share{raw};
        share_pixels(c,s,raw,color,gray) >> _both >> _norot >> _share{raw};

        share_pixels(c,s,rot) >> _color >> _rot >> _share{rot};
        share_pixels(c,s,rot,gray) >> _gray >> _rot;
        share_pixels(c,s,rot,color) >> _color >> _rot >> _share{rot};
        share_pixels(c,s,rot,color,gray) >> _both >> _rot >> _share{rot};

        // use shared image (no modify)
        FILE_LOG(logDETAIL) << "share_image";
        share_image(c,s,sraw) >> _color >> _norot >> _share{raw} >> _ptrshared{sraw};
        share_image(c,s,sraw,gray) >> _gray >> _norot >> _noshare{raw};
        share_image(c,s,sraw,color) >> _color >> _norot >> _share{raw} >> _ptrshared{sraw};
        share_image(c,s,sraw,color,gray) >> _both >> _norot >> _share{raw} >> _ptrshared{sraw};

        share_image(c,s,srot) >> _color >> _rot >> _share{rot} >> _ptrshared{srot};
        share_image(c,s,srot,gray) >> _gray >> _rot;
        share_image(c,s,srot,color) >> _color >> _rot >> _share{rot} >> _ptrshared{srot};
        share_image(c,s,srot,color,gray) >> _both >> _rot >> _share{rot} >> _ptrshared{srot};

        // move image
        FILE_LOG(logDETAIL) << "take_image";
        take_image(c,s,watchptr(copy(raw))) >> _color >> _norot >> _ptrgood;
        BOOST_CHECK(s_ptr_deleted);
        take_image(c,s,watchptr(copy(raw)),gray) >> _gray >> _norot >> _ptrgood;
        BOOST_CHECK(s_ptr_deleted);
        take_image(c,s,watchptr(copy(raw)),color) >> _color >> _norot >> _ptrgood;
        BOOST_CHECK(s_ptr_deleted);
        take_image(c,s,watchptr(copy(raw)),color,gray) >> _both >> _norot >> _ptrgood;
        BOOST_CHECK(s_ptr_deleted);

        take_image(c,s,watchptr(copy(rot))) >> _color >> _rot >> _ptrgood;
        BOOST_CHECK(s_ptr_deleted);
        take_image(c,s,watchptr(copy(rot)),gray) >> _gray >> _rot >> _ptrgood;
        BOOST_CHECK(s_ptr_deleted);
        take_image(c,s,watchptr(copy(rot)),color) >> _color >> _rot >> _ptrgood;
        BOOST_CHECK(s_ptr_deleted);
        take_image(c,s,watchptr(copy(rot)),color,gray) >> _both >> _rot >> _ptrgood;
        BOOST_CHECK(s_ptr_deleted);

        // use pixels (can modify them)
        FILE_LOG(logDETAIL) << "use_pixels";
        use_pixels(c,s,*copy(raw)) >> _color >> _norot;
        use_pixels(c,s,*copy(raw),gray) >> _gray >> _norot;
        use_pixels(c,s,*copy(raw),color) >> _color >> _norot;
        use_pixels(c,s,*copy(raw),color,gray) >> _both >> _norot;

        use_pixels(c,s,*copy(rot)) >> _color >> _rot;
        use_pixels(c,s,*copy(rot),gray) >> _gray >> _rot;
        use_pixels(c,s,*copy(rot),color) >> _color >> _rot;
        use_pixels(c,s,*copy(rot),color,gray) >> _both >> _rot;
    }

    {
        FILE_LOG(logINFO) << "image: v3";
        detection_settings s;
        s.detector_version = 3;

        /*
        FILE_LOG(logDETAIL) << "load";
        load_image(c,s,img_path,rotate(1)) >> _gray >> _norot;
        load_image(c,s,img_path,gray) >> _gray >> _norot;
        load_image(c,s,img_path,color) >> _color >> _norot;
        load_image(c,s,img_path,color,gray) >> _both >> _norot;
        */

        // copy image
        FILE_LOG(logDETAIL) << "copy_image";
        copy_image(c,s,raw) >> _gray >> _norot >> _noshare{raw};
        copy_image(c,s,raw,gray) >> _gray >> _norot >> _noshare{raw};
        copy_image(c,s,raw,color) >> _color >> _norot >> _noshare{raw};
        copy_image(c,s,raw,color,gray) >> _both >> _norot >> _noshare{raw};

        copy_image(c,s,rot) >> _norotg >> _noshare{rot};
        copy_image(c,s,rot,gray) >> _gray >> _norotg >> _noshare{rot};
        copy_image(c,s,rot,color) >> _color >> _norotg >> _noshare{rot};
        copy_image(c,s,rot,color,gray) >> _both >> _norotg >> _noshare{rot};

        // use pixels (no modify)
        share_pixels(c,s,raw) >> _color >> _norot >> _share{raw};
        share_pixels(c,s,raw,gray) >> _gray >> _norot >> _noshare{raw};
        share_pixels(c,s,raw,color) >> _color >> _norot >> _share{raw};
        share_pixels(c,s,raw,color,gray) >> _both >> _norot >> _share{raw};

        share_pixels(c,s,rot) >> _norotg >> _noshare{rot};
        share_pixels(c,s,rot,gray) >> _gray >> _norotg;
        share_pixels(c,s,rot,color) >> _color >> _norotg;
        share_pixels(c,s,rot,color,gray) >> _both >> _norotg;

        // use shared image (no modify)
        FILE_LOG(logDETAIL) << "image: share_image";
        share_image(c,s,sraw) >> _color >> _norot >> _share{raw};
        share_image(c,s,sraw,gray) >> _gray >> _norot >> _noshare{raw};
        share_image(c,s,sraw,color) >> _color >> _norot >> _share{raw};
        share_image(c,s,sraw,color,gray) >> _both >> _norot >> _share{raw};

        share_image(c,s,srot) >> _norotg >> _noshare{rot};
        share_image(c,s,srot,gray) >> _gray >> _norotg;
        share_image(c,s,srot,color) >> _color >> _norotg;
        share_image(c,s,srot,color,gray) >> _both >> _norotg;

        // move image
        take_image(c,s,copy(raw)) >> _color >> _norot;
        take_image(c,s,copy(raw),gray) >> _gray >> _norot;
        take_image(c,s,copy(raw),color) >> _color >> _norot;
        take_image(c,s,copy(raw),color,gray) >> _both >> _norot;

        take_image(c,s,copy(rot)) >> _norotg;
        take_image(c,s,copy(rot),gray) >> _gray >> _norotg;
        take_image(c,s,copy(rot),color) >> _color >> _norotg;
        take_image(c,s,copy(rot),color,gray) >> _both >> _norotg;

        // take pixels
        use_pixels(c,s,*copy(raw)) >> _color >> _norot;
        use_pixels(c,s,*copy(raw),gray) >> _gray >> _norot;
        use_pixels(c,s,*copy(raw),color) >> _color >> _norot;
        use_pixels(c,s,*copy(raw),color,gray) >> _both >> _norot;

        use_pixels(c,s,*copy(rot)) >> _norotg;
        use_pixels(c,s,*copy(rot),gray) >> _gray >> _norotg;
        use_pixels(c,s,*copy(rot),color) >> _color >> _norotg;
        use_pixels(c,s,*copy(rot),color,gray) >> _both >> _norotg;
    }

    for (unsigned v : { 4u, 5u, 6u, 7u }) {
        FILE_LOG(logINFO) << "image: v" << v;
        detection_settings s;
        s.detector_version = v;

        /*
        FILE_LOG(logDETAIL) << "load";
        load_image(c,s,img_path,rotate(1)) >> _color >> _norot;
        load_image(c,s,img_path,gray) >> _gray >> _norot;
        load_image(c,s,img_path,color) >> _color >> _norot;
        load_image(c,s,img_path,color,gray) >> _both >> _norot;
        */

        // copy image
        FILE_LOG(logDETAIL) << "copy_image";
        copy_image(c,s,raw) >> _color >> _norot >> _noshare{raw};
        copy_image(c,s,raw,gray) >> _gray >> _norot >> _noshare{raw};
        copy_image(c,s,raw,color) >> _color >> _norot >> _noshare{raw};
        copy_image(c,s,raw,color,gray) >> _both >> _norot >> _noshare{raw};

        copy_image(c,s,rot) >> _color >> _norotc >> _noshare{rot};
        copy_image(c,s,rot,gray) >> _gray >> _norotc >> _noshare{rot};
        copy_image(c,s,rot,color) >> _color >> _norotc >> _noshare{rot};
        copy_image(c,s,rot,color,gray) >> _both >> _norotc >> _noshare{rot};

        // use pixels (no modify)
        share_pixels(c,s,raw) >> _color >> _norot >> _share{raw};
        share_pixels(c,s,raw,gray) >> _gray >> _norot >> _noshare{raw};
        share_pixels(c,s,raw,color) >> _color >> _norot >> _share{raw};
        share_pixels(c,s,raw,color,gray) >> _both >> _norot >> _share{raw};

        share_pixels(c,s,rot) >> _color >> _norotc >> _noshare{rot};
        share_pixels(c,s,rot,gray) >> _gray >> _norotc;
        share_pixels(c,s,rot,color) >> _color >> _norotc;
        share_pixels(c,s,rot,color,gray) >> _both >> _norotc;

        // use shared image (no modify)
        share_image(c,s,sraw) >> _color >> _norot >> _share{raw} >> _ptrshared{sraw};
        share_image(c,s,sraw,gray) >> _gray >> _norot >> _noshare{raw};
        share_image(c,s,sraw,color) >> _color >> _norot >> _share{raw} >> _ptrshared{sraw};
        share_image(c,s,sraw,color,gray) >> _both >> _norot >> _share{raw} >> _ptrshared{sraw};

        share_image(c,s,srot) >> _color >> _norotc >> _noshare{rot};
        share_image(c,s,srot,gray) >> _gray >> _norotc;
        share_image(c,s,srot,color) >> _color >> _norotc;
        share_image(c,s,srot,color,gray) >> _both >> _norotc;

        // move image
        take_image(c,s,copy(raw)) >> _color >> _norot;
        take_image(c,s,copy(raw),gray) >> _gray >> _norot;
        take_image(c,s,copy(raw),color) >> _color >> _norot;
        take_image(c,s,copy(raw),color,gray) >> _both >> _norot;

        take_image(c,s,copy(rot)) >> _color >> _norotc;
        take_image(c,s,copy(rot),gray) >> _gray >> _norotc;
        take_image(c,s,copy(rot),color) >> _color >> _norotc;
        take_image(c,s,copy(rot),color,gray) >> _both >> _norotc;

        // take pixels
        use_pixels(c,s,*copy(raw)) >> _color >> _norot;
        use_pixels(c,s,*copy(raw),gray) >> _gray >> _norot;
        use_pixels(c,s,*copy(raw),color) >> _color >> _norot;
        use_pixels(c,s,*copy(raw),color,gray) >> _both >> _norot;

        use_pixels(c,s,*copy(rot)) >> _color >> _norotc;
        use_pixels(c,s,*copy(rot),gray) >> _gray >> _norotc;
        use_pixels(c,s,*copy(rot),color) >> _color >> _norotc;
        use_pixels(c,s,*copy(rot),color,gray) >> _both >> _norotc;
    }

    BOOST_CHECK(raw_image_same_pixels(*raw, *raw_orig));
    BOOST_CHECK(raw_image_same_pixels(*rot, *rot_orig));

    {
        FILE_LOG(logINFO) << "image: modify";
        detection_settings s;
        s.detector_version = 3;

        {
            FILE_LOG(logDETAIL) << "take_image";
            auto rot_copy = copy(rot);
            auto rot_ptr = rot_copy.get();
            const auto i = take_image(c,s,move(rot_copy));
            i >> _share{rot_ptr};
            BOOST_CHECK(!raw_image_same_meta(*rot_ptr, *rot_orig));
        }

        FILE_LOG(logDETAIL) << "use_pixels";
        use_pixels(c,s,*rot) >> _share{rot};
        BOOST_CHECK(raw_image_same_meta(*rot, *rot_orig));
        BOOST_CHECK(!raw_image_same_pixels(*rot, *rot_orig));
    }
    
    FILE_LOG(logINFO) << "image: done";
}

BOOST_AUTO_TEST_SUITE_END()
