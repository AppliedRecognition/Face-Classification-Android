
#include <dlibx/net_vector.hpp>
#include <dlibx/rotated_box.hpp>

#include <dlib/image_transforms/interpolation.h>

#include "extract.hpp"
#include "model.hpp"

#include <rec/internal_prototype_1.hpp>
#include <rec/fpvc.hpp>

#include <det/types.hpp>
#include <core/thread_data.hpp>

#include <dlibx/raw_image.hpp>
#include <raw_image/reader.hpp>
#include <raw_image/point_rounding.hpp>
#include <raw_image/adjust.hpp>
#include <raw_image/input_extractor.hpp>

#include <applog/core.hpp>

#include <numeric>


using rec::version_type;

static const auto k_brightness = json::string("brightness");
static const auto k_contrast = json::string("contrast");
static const auto k_min_contrast = json::string("min_contrast");

static std::vector<int> signed_vector(const json::value& val) {
    std::vector<int> vec;
    if (json::is_type<json::array>(val)) {
        for (auto&& x : get_array(val)) {
            if (!json::is_type<json::integer>(x))
                throw std::invalid_argument("expected integer or array of integers");
            auto i = get_integer(x);
            if (!(-256 < i && i < 256))
                throw std::invalid_argument("integer out of range");
            if (i != 0)
                vec.push_back(int(i));
        }
    }
    else if (json::is_type<json::integer>(val)) {
        auto i = get_integer(val);
        if (!(-256 < i && i < 256))
            throw std::invalid_argument("integer out of range");
        if (i != 0)
            vec.push_back(int(i));
    }
    else if (val != json::null)
        throw std::invalid_argument("expected integer or array of integers");
    sort(vec.begin(), vec.end());
    vec.erase(unique(vec.begin(), vec.end()), vec.end());
    return vec;
}

static std::vector<int> unsigned_vector(const json::value& val) {
    std::vector<int> vec;
    if (json::is_type<json::array>(val)) {
        for (auto&& x : get_array(val)) {
            if (!json::is_type<json::integer>(x))
                throw std::invalid_argument("expected integer or array of integers");
            auto i = get_integer(x);
            if (!(-256 < i && i < 256))
                throw std::invalid_argument("integer out of range");
            if (i != 0)
                vec.push_back(int(std::abs(i)));
        }
    }
    else if (json::is_type<json::integer>(val)) {
        auto i = get_integer(val);
        if (!(-256 < i && i < 256))
            throw std::invalid_argument("integer out of range");
        if (i != 0)
            vec.push_back(int(std::abs(i)));
    }
    else if (val != json::null)
        throw std::invalid_argument("expected integer or array of integers");
    sort(vec.begin(), vec.end());
    vec.erase(unique(vec.begin(), vec.end()), vec.end());
    return vec;
}

static auto cb_measure(const raw_image::plane& yuv) {
    assert(yuv.layout == raw_image::pixel::yuv);
    unsigned long sum = 0, ss = 0;
    for (auto&& line : raw_image::read_lines_bpp<3>(yuv))
        for (auto&& pixel : line) {
            sum += pixel[0];
            ss += pixel[0]*unsigned(pixel[0]);
        }
    const auto n = std::size_t(yuv.width) * yuv.height;
    const auto mean = int((sum + n/2) / n);
    const auto var = std::max(1, int((ss + n/2) / n) - mean*mean);
    return std::pair(mean,int(std::lround(std::sqrt(var))));
}

static auto cb_adjust(const raw_image::plane& yuv,
                      const std::pair<int,int>& before, int after_stddev,
                      raw_image::pixel_layout dest_layout) {
    assert(yuv.layout == raw_image::pixel::yuv);
    auto quad_convert =
        [=](uint8_t* dest, const uint8_t* src, unsigned nquads) {
            for ( ; nquads > 0; --nquads) {
                for (auto n = 4; n > 0; --n, dest += 3, src += 3) {
                    const auto z = 128 + (src[0] - before.first) * after_stddev / before.second;
                    dest[0] =
                        static_cast<uint8_t>(z < 0 ? 0 : z < 256 ? z : 255);
                    dest[1] = src[1];
                    dest[2] = src[2];
                }
            }
        };
    return copy(convert(transform_quads(raw_image::reader::construct(yuv),
                                        yuv.layout, quad_convert),
                        dest_layout));
}

static auto cd_scale(dlib::chip_details cd, float factor) {
    const auto cx = (cd.rect.left() + cd.rect.right() + 1) / 2;
    const auto cy = (cd.rect.top() + cd.rect.bottom() + 1) / 2;
    const auto w = factor * (1 + cd.rect.right() - cd.rect.left());
    const auto h = factor * (1 + cd.rect.bottom() - cd.rect.top());
    cd.rect.left()   = cx - w/2;
    cd.rect.right()  = cx + w/2 - 1;
    cd.rect.top()    = cy - h/2;
    cd.rect.bottom() = cy + h/2 - 1;
    return cd;
}

// width is expanded by 2*horz and height by 2*vert.
static auto cd_expand(dlib::chip_details cd, unsigned horz, unsigned vert) {
    const auto nc = cd.cols + 2*horz;
    const auto nr = cd.rows + 2*vert;
    const auto dw = (cd.rect.width()  * double(nc) / double(cd.cols) - cd.rect.width())  / 2;
    const auto dh = (cd.rect.height() * double(nr) / double(cd.rows) - cd.rect.height()) / 2;
    cd.rect.left()   -= dw;
    cd.rect.right()  += dw;
    cd.rect.top()    -= dh;
    cd.rect.bottom() += dh;
    cd.cols = nc;
    cd.rows = nr;
    return cd;
}


static auto get_chip_details(const det::face_coordinates& coordinates,
                             const dlibx::net::vector& net) {
    if (!net.input_extractor)
        throw std::runtime_error("recognition model has no input extractor");
    auto& extractor = *net.input_extractor;

    // extract face chip
    const det::detected_coordinates* dcp = nullptr;
    for (const auto& dc : coordinates)
        if (dc.landmarks.size() > 2)
            dcp = &dc;
    if (!dcp) {
        std::stringstream ss;
        ss << "dlib template extraction requires landmarks";
        for (const auto& dc : coordinates) {
            const auto x = stdx::round((dc.eye_left.x+dc.eye_right.x)/2);
            const auto y = stdx::round((dc.eye_left.y+dc.eye_right.y)/2);
            ss << " (" << int(dc.type)
               << ',' << dc.landmarks.size()
               << ',' << x << ',' << y << ')';
        }
        FILE_LOG(logERROR) << ss.str();
        throw std::logic_error(
            "dlib template extraction requires landmarks");
    }
    std::vector<raw_image::point2f> pts;
    pts.reserve(dcp->landmarks.size());
    for (auto&& p : dcp->landmarks)
        pts.push_back(raw_image::round_from(p));
    return std::make_pair(extractor(pts), extractor.layout);
}

std::vector<rec::prototype_ptr>
rec::dlib::jitter(const raw_image::multi_plane_arg& image,
                  const det::face_coordinates& coordinates,
                  version_type ver,
                  const json::object& options,
                  core::thread_data& td) {

    auto&& net_pair = core::get<thread_map>(td.thread).get(ver, td);
    auto& net = *net_pair.first;

    // options
    const auto roll = unsigned_vector(options["roll"]);
    const auto horz = unsigned_vector(options["horz"]);
    const auto vert = signed_vector(options["vert"]);
    const auto scale = signed_vector(options["scale"]);
    const auto contrast = signed_vector(options[k_contrast]);
    const auto cbase = make_number(options["cbase"],48);
    if (cbase < 1 || 255 < cbase)
        throw std::invalid_argument("contrast base 'cbase' out of range");

    // chip details and central image
    const auto cdp = get_chip_details(coordinates, net);
    using px = raw_image::pixel;
    const auto central = extract_image_chip(image, cdp.first, px::yuv);
    const auto cb = cb_measure(*central);

    // chips
    std::vector<raw_image::plane_ptr> chips;
    chips.push_back(cb_adjust(*central, cb, cbase, cdp.second));

    // roll
    for (auto degrees : roll) {
        for (auto i : { -1.f, 1.f }) {
            auto adj = cdp.first;
            adj.angle += i * float(degrees * M_PI / 180);
            auto chip = extract_image_chip(image, adj, px::yuv);
            chips.emplace_back(cb_adjust(*chip, cb, cbase, cdp.second));
        }
    }

    // horz and vert
    int h_extra = 0, v_extra = 0;
    for (auto i : horz) {
        assert(i > 0);
        h_extra = std::max(h_extra, i);
    }
    for (auto i : vert)
        v_extra = std::max(v_extra, std::abs(i));
    if (h_extra > 0 || v_extra > 0) {
        auto adj = cd_expand(cdp.first, unsigned(h_extra), unsigned(v_extra));
        auto area = extract_image_chip(image, adj, px::yuv);
        for (auto x : horz) {
            auto c0 = crop(area, unsigned(h_extra-x), unsigned(v_extra),
                           central->width, central->height);
            auto c1 = crop(area, unsigned(h_extra+x), unsigned(v_extra),
                           central->width, central->height);
            chips.emplace_back(cb_adjust(c0, cb, cbase, cdp.second));
            chips.emplace_back(cb_adjust(c1, cb, cbase, cdp.second));
        }
        for (auto y : vert) {
            auto chip = crop(area, unsigned(h_extra), unsigned(v_extra+y),
                             central->width, central->height);
            chips.emplace_back(cb_adjust(chip, cb, cbase, cdp.second));
        }
    }

    // scale
    for (auto e : scale) {
        auto adj = cd_scale(cdp.first, std::exp(float(e) / 64.0f));
        auto chip = extract_image_chip(image, adj, px::yuv);
        chips.emplace_back(cb_adjust(*chip, cb, cbase, cdp.second));
    }

    // contrast
    for (auto delta : contrast) {
        const auto c = std::max(1, cbase + delta);
        chips.emplace_back(cb_adjust(*central, cb, c, cdp.second));
    }

    // extract vectors
    std::vector<std::vector<float> > desc;
    desc.reserve(chips.size());
    net(chips.begin(), chips.end(), desc);
    assert(desc.size() == chips.size());

    // assemble prototypes
    std::vector<rec::prototype_ptr> r;
    r.reserve(desc.size());
    for (std::size_t i = 0; i < desc.size(); ++i) {
        auto v8 = internal::fpvc_vector_compress(desc[i].begin(),desc[i].end());
        auto proto = internal::prototype_1::make_shared(net_pair.second, move(v8));
        proto->thumb = move(chips[i]);
        r.emplace_back(move(proto));
    }
    return r;
}

static bool warn_no_color = false;

rec::rotated_box
rec::dlib::bounding_box(const det::face_coordinates& coordinates,
                        version_type ver,
                        const core::context_data& cd) {
    const auto model_ptr = core::get<context_map>(cd.context).load(ver, &load_shared, ver, cd).first;
    if (!model_ptr) throw std::runtime_error("failed to load model");
    return to_rotated_box(get_chip_details(coordinates, *model_ptr).first);
}

static auto apply_options(
    const raw_image::plane& chip, const json::object& options) {

    // target brightness
    std::optional<std::array<float,2> > tb;
    auto& vb = options[k_brightness];
    if (json::is_type<json::array>(vb))
        tb = json::make_array<float,2>(vb);
    else if (vb != json::null) {
        auto x = json::make_number<float>(vb);
        tb = {x,x};
    }
    if (tb && !(0 <= (*tb)[0] && (*tb)[0] <= (*tb)[1])) {
        FILE_LOG(logERROR) << "invalid brightness values: min " << (*tb)[0]
                           << " max " << (*tb)[1];
        throw std::invalid_argument("invalid brightness values");
    }

    // target contrast
    std::optional<std::array<float,2> > tc;
    auto& vc = options[k_contrast];
    auto& vmc = options[k_min_contrast];
    if (vmc != json::null) {
        if (vc != json::null)
            throw std::invalid_argument(
                "cannot specify both contrast and min_contrast");
        auto x = json::make_number<float>(vmc);
        tc = {x,256};
    }
    else if (json::is_type<json::array>(vc))
        tc = json::make_array<float,2>(vc);
    else if (vc != json::null) {
        auto x = json::make_number<float>(vc);
        tc = {x,x};
    }
    if (tc && !(0 <= (*tc)[0] && (*tc)[0] <= (*tc)[1])) {
        FILE_LOG(logERROR) << "invalid contrast values: min " << (*tc)[0]
                           << " max " << (*tc)[1];
        throw std::invalid_argument("invalid contrast values");
    }

    // use the center approx 50% by pixels of the chip
    // note 5/7 ~= 1/sqrt(2) so we want borders of 1/7
    const auto x = chip.width / 7;
    const auto y = chip.height / 7;

    if (tc) {
        auto bc = measure_brightness_contrast(
            crop(chip, x, y, chip.width-2*x, chip.height-2*y));
        float alpha = 1;
        if (bc.contrast < (*tc)[0])
            alpha = (*tc)[0] / std::max(1.0f, bc.contrast);
        else if ((*tc)[1] < bc.contrast)
            alpha = (*tc)[1] / std::max(1.0f, bc.contrast);
        float beta = bc.brightness * (1 - alpha); // no change in brightness
        if (tb) {
            if (bc.brightness < (*tb)[0])
                beta = (*tb)[0] - bc.brightness * alpha;
            else if ((*tb)[1] < bc.brightness)
                beta = (*tb)[1] - bc.brightness * alpha;
        }
        in_place_linear_adjust(chip, alpha, beta);
    }
    else if (tb) {
        const auto b = measure_brightness(
            crop(chip, x, y, chip.width-2*x, chip.height-2*y));
        if (b < (*tb)[0])
            in_place_linear_adjust(chip, 1.0f, (*tb)[0] - b);
        else if ((*tb)[1] < b)
            in_place_linear_adjust(chip, 1.0f, (*tb)[1] - b);
    }
}

rec::prototype_ptr
rec::dlib::extract(const raw_image::multi_plane_arg& image,
                   const rotated_box& rbox,
                   version_type ver,
                   const json::object& options,
                   core::thread_data& td) {

    if (image.size() == 1 &&
        bytes_per_pixel(image.front().layout) == 1 &&
        !warn_no_color) {
        warn_no_color = true;
        FILE_LOG(logWARNING) << "rec: grayscale image used to extract template";
    }

    auto&& net_pair = core::get<thread_map>(td.thread).get(ver, td);
    auto& net = *net_pair.first;

    // restore chip_details from rotated_box
    if (!net.input_extractor)
        throw std::runtime_error("recognition model has no input extractor");
    auto& extractor = *net.input_extractor;
    auto chip = to_chip_details(rbox);
    chip.rows = extractor.width;
    chip.cols = extractor.height;

    // extract face chip
    auto face_chip = extract_image_chip(image, chip, extractor.layout);
    apply_options(*face_chip, options);

    // neural net: face chip -> vector
    std::vector<float> desc;
    net(face_chip,desc);

    auto v8 = internal::fpvc_vector_compress(desc.begin(),desc.end());
    auto proto = internal::prototype_1::make_shared(move(net_pair.second), move(v8));
    proto->thumb = move(face_chip);
    return proto;
}

rec::prototype_ptr
rec::dlib::extract(const raw_image::multi_plane_arg& image,
                   const det::face_coordinates& coordinates,
                   version_type ver,
                   const json::object& options,
                   core::thread_data& td) {

    if (image.size() == 1 &&
        bytes_per_pixel(image.front().layout) == 1 &&
        !warn_no_color) {
        warn_no_color = true;
        FILE_LOG(logWARNING) << "rec: grayscale image used to extract template";
    }

    auto&& net_pair = core::get<thread_map>(td.thread).get(ver, td);
    auto& net = *net_pair.first;

    // extract face chip
    const auto cdp = get_chip_details(coordinates, net);
    auto face_chip = extract_image_chip(image, cdp.first, cdp.second);
    apply_options(*face_chip, options);

    // neural net: face chip -> vector
    std::vector<float> desc;
    net(face_chip,desc);

    auto v8 = internal::fpvc_vector_compress(desc.begin(),desc.end());
    auto proto = internal::prototype_1::make_shared(move(net_pair.second), move(v8));
    proto->thumb = move(face_chip);
    return proto;
}

rec::prototype_ptr
rec::dlib::from_face_chip(raw_image::plane_ptr face_chip,
                          version_type ver,
                          core::thread_data& td) {

    auto&& net_pair = core::get<thread_map>(td.thread).get(ver, td);
    auto& net = *net_pair.first;

    // neural net: face chip -> vector
    std::vector<float> desc;
    net(face_chip,desc);

    auto v8 = internal::fpvc_vector_compress(desc.begin(),desc.end());
    auto proto = internal::prototype_1::make_shared(move(net_pair.second), move(v8));
    proto->thumb = move(face_chip);
    return proto;
}
