#include <boost/test/unit_test.hpp>

#include <rec/internal_prototype_1.hpp>
#include <rec/model.hpp>

#include <applog/core.hpp>
#include <random>

using namespace rec::internal;
using prototype = rec::internal::prototype_1;
using rec::internal::fpvc_vector_type;

namespace stdx {
    template <typename T>
    bool operator==(const span<T>& a, const span<T>& b) {
        return a.size() == b.size() &&
            std::equal(a.begin(), a.end(), b.begin());
    }
}

/*
static bool operator==(const fp16vec& a, const fp16vec& b) {
    return std::abs(a.coeff - b.coeff) <= 0 && a.size() == b.size() &&
        std::equal(a.begin(), a.end(), b.begin());
}
*/

BOOST_AUTO_TEST_SUITE(rec_dlib)

BOOST_AUTO_TEST_CASE(prototype_serialize) {
    std::mt19937 gen(1);

    std::vector<uint8_t> nums;
    for (unsigned i = 0; i < 256; ++i)
        nums.emplace_back(i);
    shuffle(nums.begin(), nums.end(), gen);
    BOOST_REQUIRE(nums[0] != 0);

    // 103 and 157 are relatively prime to all of fpvc_s16_decompress_table
    static constexpr auto denom = 1024*1024;

    const auto model =
        std::make_shared<rec::model_state>(
            rec::model_static{16,rec::variant::cos,5,0,0,nullptr});

    fpvc_vector_type v0;
    v0.first = 103.0f / denom;
    v0.second.assign(nums.begin(), nums.begin() + 128);
    const auto p0 = prototype_1::make_shared(model, move(v0));
    BOOST_CHECK_EQUAL(std::get<1>(p0->get16()), std::get<1>(p0->get8()));
    
    fpvc_vector_type v1;
    v1.first = 157.0f / denom;
    v1.second.assign(nums.begin() + 128, nums.begin() + 256);
    const auto p1 = prototype_1::make_shared(model, move(v1));
    BOOST_CHECK_EQUAL(std::get<1>(p1->get16()), std::get<1>(p1->get8()));

    BOOST_CHECK_EQUAL(p0->uuid.size(), 16);
    BOOST_CHECK_EQUAL(p1->uuid.size(), 16);
    BOOST_CHECK(p0->uuid != p1->uuid);

    const auto p0f0 = p0->serialize();
    const auto p1f0 = p1->serialize();
    model->serialize_format.store(1);
    const auto p0f1 = p0->serialize();
    const auto p1f1 = p1->serialize();
    model->serialize_format.store(2);
    const auto p0f2 = p0->serialize();
    const auto p1f2 = p1->serialize();
    model->serialize_format.store(3);
    const auto p0f3 = p0->serialize();
    const auto p1f3 = p1->serialize();
    model->serialize_format.store(0);

    const auto check =
        [&](auto& p, auto& pf0, auto& pf1, auto& pf2, auto& pf3) {
        BOOST_CHECK_EQUAL(pf0.size(), 140);  // 4 + 8 + 128
        BOOST_CHECK_EQUAL(pf1.size(), 132);  // 4     + 128
        BOOST_CHECK_EQUAL(pf2.size(), 200);  // 4 + 4 + 192
        BOOST_CHECK_EQUAL(pf3.size(), 264);  // 4 + 4 + 256
        
        const auto pd0 =
            std::static_pointer_cast<const prototype_1>(
                prototype_1::deserialize(model, pf0.data(), pf0.size()));
        const auto pd1 =
            std::static_pointer_cast<const prototype_1>(
                prototype_1::deserialize(model, pf1.data(), pf1.size()));
        const auto pd2 =
            std::static_pointer_cast<const prototype_1>(
                prototype_1::deserialize(model, pf2.data(), pf2.size()));
        const auto pd3 =
            std::static_pointer_cast<const prototype_1>(
                prototype_1::deserialize(model, pf3.data(), pf3.size()));
        
        BOOST_CHECK(p->uuid == pd0->uuid);
        BOOST_CHECK(p->uuid != pd1->uuid);
        BOOST_CHECK(pd1->uuid != pd2->uuid);
        BOOST_CHECK(p->uuid != pd2->uuid);
        BOOST_CHECK(pd2->uuid == pd3->uuid);

        auto p16 = p->get16();
        for (auto* pd : { pd0.get(), pd2.get(), pd3.get() })
            BOOST_CHECK(p16 == pd->get16());

        BOOST_CHECK(p->get8() == pd0->get8());
        for (auto* pd : { pd1.get(), pd2.get(), pd3.get() })
            BOOST_CHECK(pd->get8().first.empty());

        const auto pd0f0 = pd0->serialize();
        const auto pd1f0 = pd1->serialize();
        const auto pd2f0 = pd2->serialize();
        BOOST_CHECK(pd0f0 == pf0);
        BOOST_CHECK(pd1f0 == pf1);
        BOOST_CHECK(pd2f0 == pf2);

        const auto pr0 = pd0->get32_orig();
        const auto pr1 = pd1->get32_orig();
        std::vector<float> f32_0(pr0.first, std::next(pr0.first, pr0.second));
        std::vector<float> f32_1(pr1.first, std::next(pr1.first, pr1.second));
        BOOST_REQUIRE_EQUAL(f32_0.size(), f32_1.size());
        float err = 0, maxabs = 0;
        for (unsigned i = 0; i < f32_0.size(); ++i) {
            err = std::max(err, std::abs(f32_0[i] - f32_1[i]));
            maxabs = std::max(maxabs, std::abs(f32_0[i]));
            maxabs = std::max(maxabs, std::abs(f32_1[i]));
        }
        BOOST_CHECK(err / maxabs < 0.012);
    };

    check(p0, p0f0, p0f1, p0f2, p0f3);
    check(p1, p1f0, p1f1, p1f2, p1f3);
}

BOOST_AUTO_TEST_SUITE_END()
