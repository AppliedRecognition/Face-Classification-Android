
#include "frvt11.h"

#include <rec/prototype.hpp>
#include <rec/multiface.hpp>
#include <rec_ncnn/engine.hpp>

#include <det_ncnn/init.hpp>
#include <det/image.hpp>

#include <core/context.hpp>

#include <json/io.hpp>

#include <raw_image/core.hpp>

#include <cmath>
#include <filesystem>


namespace FRVT {
    inline auto to_raw_image(const Image& img) {
        raw_image::plane p;
        p.data = img.data.get();
        switch (img.depth) {
        case 8:  p.layout = raw_image::pixel::gray8; break;
        case 24: p.layout = raw_image::pixel::rgb24; break;
        default:
            throw std::invalid_argument("image depth must be 8 or 24");
        }
        p.height = img.height;
        p.width = img.width;
        p.bytes_per_line = img.width * unsigned(img.depth / 8);
        return p;
    }
}

namespace det {
    inline auto to_EyePair(const eye_coordinates& ec,
                           unsigned width, unsigned height) {
        FRVT::EyePair ep;
        const auto lx = std::lround(ec.eye_left.x);
        const auto ly = std::lround(ec.eye_left.y);
        const auto rx = std::lround(ec.eye_right.x);
        const auto ry = std::lround(ec.eye_right.y);
        if (0 <= lx && lx < width &&
            0 <= ly && ly < height) {
            ep.isLeftAssigned = true;
            ep.xleft = uint16_t(lx);
            ep.yleft = uint16_t(ly);
        }
        if (0 <= rx && rx < width &&
            0 <= ry && ry < height) {
            ep.isRightAssigned = true;
            ep.xright = uint16_t(rx);
            ep.yright = uint16_t(ry);
        }
        return ep;
    }
}

namespace {
    class tester : public FRVT_11::Interface {
        static constexpr auto rec_version = rec::version_type{24};

        core::context_ptr context;
        det::detection_settings ds;

        static auto sizequal(const det::face_coordinates& face) {
            if (face.empty()) return 0.0f;
            auto& dc = face.back();
            if (dc.landmarks.size() != 68) return 0.0f;
            return dc.confidence * dc.eye_distance();
        }

    public:
        using ReturnStatus = FRVT::ReturnStatus;
        using ReturnCode = FRVT::ReturnCode;

        ReturnStatus
        initialize(const std::string& configDir) override {
            const auto path = std::filesystem::path(configDir);
            const auto dsv = json::load(path / "settings.json");
            ds.assign(get_object(dsv));
            core::context_settings cs;
            cs.min_threads = cs.max_threads = 1; // single thread
            context = core::context::construct(cs);
            det::ncnn::init(context);
            det::prepare_detection(context, ds, path);
            rec::ncnn::initialize(context, path);
            rec::prototype::load_model(context, rec_version);
            rec::prototype::set_serialize_format(context, rec_version, 1);
            return ReturnCode::Success;
        }

        ReturnStatus
        matchTemplates(
            const std::vector<uint8_t>& verifTemplate,
            const std::vector<uint8_t>& enrollTemplate,
            double& score) override {

            if (verifTemplate.size() == 132 &&
                enrollTemplate.size() == 132) {
                const auto* a =
                    reinterpret_cast<const int8_t*>(verifTemplate.data()) + 4;
                const auto* b =
                    reinterpret_cast<const int8_t*>(enrollTemplate.data()) + 4;
                int sum = 0, as = 0, bs = 0;
                for (auto n = 128; n > 0; --n, ++a, ++b) {
                    as += *a**a;
                    bs += *b**b;
                    sum += *a**b;
                }
                if (0 < as && 0 < bs) {
                    score = 1 + double(sum) / std::sqrt(double(as)*double(bs));
                    return ReturnCode::Success;
                }
            }
            else if (!verifTemplate.empty() && !enrollTemplate.empty()) {
                auto a = rec::prototype::deserialize(
                    context, verifTemplate.data(), verifTemplate.size());
                auto b = rec::multiface(
                    context, stdx::binary(enrollTemplate.data(),
                                          enrollTemplate.size()));
                score = 1 + compare(a, b, rec::variant::raw);
                return ReturnCode::Success;
            }
            // else at least one template is empty
            score = -1;
            return ReturnCode::VerifTemplateError;
        }

        // single image with multiple faces -> multiple templates
        ReturnStatus
        createFaceTemplate(
            const FRVT::Image& image, FRVT::TemplateRole,
            std::vector<std::vector<uint8_t>>& templs,
            std::vector<FRVT::EyePair>& eyeCoordinates) override {
            const auto raw = to_raw_image(image);
            const auto found =
                det::detect_faces(context, ds, share_pixels(context, ds, raw));
            if (found.empty()) {
                templs.emplace_back();
                eyeCoordinates.emplace_back();
                return ReturnCode::FaceDetectionError;
            }
            templs.reserve(found.size());
            eyeCoordinates.reserve(found.size());
            for (auto& fc : found) {
                const auto proto = rec::extract(context, raw, fc, rec_version);
                const auto bin = to_binary(proto);
                const auto data = bin.data<uint8_t>();
                templs.emplace_back(data, data + bin.size());
                auto& dc = fc.back();
                eyeCoordinates.emplace_back(
                    to_EyePair(dc,raw.width,raw.height));
            }
            return ReturnCode::Success;
        }

        // multiple images with one face each -> single template
        ReturnStatus
        createFaceTemplate(
            const std::vector<FRVT::Image>& images, FRVT::TemplateRole,
            std::vector<uint8_t>& templ,
            std::vector<FRVT::EyePair>& eyeCoordinates) override {
            eyeCoordinates.reserve(images.size());
            std::vector<rec::prototype_ptr> protos;
            protos.reserve(images.size());
            for (auto& image : images) {
                const auto raw = to_raw_image(image);
                const auto found =
                    det::detect_faces(context, ds,
                                      share_pixels(context, ds, raw));
                auto best = found.begin();
                for (auto it = best, end = found.end(); it != end; ++it)
                    if (sizequal(*best) < sizequal(*it))
                        best = it;
                if (best != found.end()) {
                    auto& fc = *best;
                    auto& dc = fc.back();
                    eyeCoordinates.emplace_back(
                        to_EyePair(dc,raw.width,raw.height));
                    protos.emplace_back(
                        rec::extract(context, raw, fc, rec_version));
                }
                else
                    eyeCoordinates.emplace_back();
            }
            if (1 < protos.size()) {
                auto mf = rec::multiface(protos.begin(), protos.end());
                auto bin = to_binary(mf);
                const auto data = bin.data<uint8_t>();
                templ.assign(data, data + bin.size());
                return ReturnCode::Success;
            }
            else if (protos.size() == 1) {
                auto bin = to_binary(protos.front());
                const auto data = bin.data<uint8_t>();
                templ.assign(data, data + bin.size());
                return ReturnCode::Success;
            }
            else
                return ReturnCode::FaceDetectionError;
        }

        ReturnStatus
        createIrisTemplate(
            const std::vector<FRVT::Image>&, FRVT::TemplateRole,
            std::vector<uint8_t>&, std::vector<FRVT::IrisAnnulus>&) override {
            return ReturnCode::NotImplemented;
        }
    };
}

std::shared_ptr<FRVT_11::Interface>
FRVT_11::Interface::getImplementation() {
    return std::make_shared<tester>();
}
