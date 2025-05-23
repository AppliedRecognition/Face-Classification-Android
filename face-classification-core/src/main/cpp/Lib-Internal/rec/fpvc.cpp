
#include "fpvc.hpp"
#include "internal_serialize.hpp"

#include <applog/core.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>


using namespace rec;
using namespace rec::internal;


static const std::pair<unsigned short, unsigned short> stride_table[] = {
    {  1,   1 },
    {  3,   1 },
    {  4,   1 },
    {  8,  12 },
    {  9,  10 },
    { 10,   8 },
    { 11,   6 },
    { 12,   5 },
    { 13,   4 },
    { 14,   3 },
    { 15,   2 },
    { 16,   0 },  // second == 0 indicates end of table
};

unsigned internal::fpvc_unsigned_compress(int xi) {
    if (xi <= 0) return 0;
    auto x = unsigned(xi);
    unsigned result = 0;
    for (const auto& p : stride_table) {
        const auto limit = p.first * unsigned(p.second);
        if (x < limit || p.second == 0)
            return result + (x + p.first/2) / p.first;
        x -= limit;
        result += p.second;
    }
    AR_ASSERT(!"end of stride_table not found");
}

int internal::fpvc_unsigned_decompress(unsigned x) {
    unsigned result = 0;
    for (const auto& p : stride_table) {
        if (x < p.second || p.second == 0) {
            const auto i = int(result + x * p.first);
            assert(i >= 0);
            return i;
        }
        x -= p.second;
        result += p.first * unsigned(p.second);
    }
    AR_ASSERT(!"end of stride_table not found");
}

const int16_t internal::fpvc_s16_decompress_table[256] = {
    0, 1, 4, 8, 16, 24, 32, 40, 
    48, 56, 64, 72, 80, 88, 96, 104, 
    113, 122, 131, 140, 149, 158, 167, 176, 
    185, 194, 204, 214, 224, 234, 244, 254, 
    264, 274, 285, 296, 307, 318, 329, 340, 
    352, 364, 376, 388, 400, 413, 426, 439, 
    452, 466, 480, 494, 509, 524, 540, 556, 
    572, 588, 604, 620, 636, 652, 668, 684, 
    700, 716, 732, 748, 764, 780, 796, 812, 
    828, 844, 860, 876, 892, 908, 924, 940, 
    956, 972, 988, 1004, 1020, 1036, 1052, 1068, 
    1084, 1100, 1116, 1132, 1148, 1164, 1180, 1196, 
    1212, 1228, 1244, 1260, 1276, 1292, 1308, 1324, 
    1340, 1356, 1372, 1388, 1404, 1420, 1436, 1452, 
    1468, 1484, 1500, 1516, 1532, 1548, 1564, 1580, 
    1596, 1612, 1628, 1644, 1660, 1676, 1692, 1708, 
    -1708, -1692, -1676, -1660, -1644, -1628, -1612, -1596, 
    -1580, -1564, -1548, -1532, -1516, -1500, -1484, -1468, 
    -1452, -1436, -1420, -1404, -1388, -1372, -1356, -1340, 
    -1324, -1308, -1292, -1276, -1260, -1244, -1228, -1212, 
    -1196, -1180, -1164, -1148, -1132, -1116, -1100, -1084, 
    -1068, -1052, -1036, -1020, -1004, -988, -972, -956, 
    -940, -924, -908, -892, -876, -860, -844, -828, 
    -812, -796, -780, -764, -748, -732, -716, -700, 
    -684, -668, -652, -636, -620, -604, -588, -572, 
    -556, -540, -524, -509, -494, -480, -466, -452, 
    -439, -426, -413, -400, -388, -376, -364, -352, 
    -340, -329, -318, -307, -296, -285, -274, -264, 
    -254, -244, -234, -224, -214, -204, -194, -185, 
    -176, -167, -158, -149, -140, -131, -122, -113, 
    -104, -96, -88, -80, -72, -64, -56, -48, 
    -40, -32, -24, -16, -8, -4, -1, 0
};

const float internal::fpvc_f32_decompress_table[256] = {
    0.0f, 1.0f, 4.0f, 8.0f, 16.0f, 24.0f, 32.0f, 40.0f, 
    48.0f, 56.0f, 64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 
    113.0f, 122.0f, 131.0f, 140.0f, 149.0f, 158.0f, 167.0f, 176.0f, 
    185.0f, 194.0f, 204.0f, 214.0f, 224.0f, 234.0f, 244.0f, 254.0f, 
    264.0f, 274.0f, 285.0f, 296.0f, 307.0f, 318.0f, 329.0f, 340.0f, 
    352.0f, 364.0f, 376.0f, 388.0f, 400.0f, 413.0f, 426.0f, 439.0f, 
    452.0f, 466.0f, 480.0f, 494.0f, 509.0f, 524.0f, 540.0f, 556.0f, 
    572.0f, 588.0f, 604.0f, 620.0f, 636.0f, 652.0f, 668.0f, 684.0f, 
    700.0f, 716.0f, 732.0f, 748.0f, 764.0f, 780.0f, 796.0f, 812.0f, 
    828.0f, 844.0f, 860.0f, 876.0f, 892.0f, 908.0f, 924.0f, 940.0f, 
    956.0f, 972.0f, 988.0f, 1004.0f, 1020.0f, 1036.0f, 1052.0f, 1068.0f, 
    1084.0f, 1100.0f, 1116.0f, 1132.0f, 1148.0f, 1164.0f, 1180.0f, 1196.0f, 
    1212.0f, 1228.0f, 1244.0f, 1260.0f, 1276.0f, 1292.0f, 1308.0f, 1324.0f, 
    1340.0f, 1356.0f, 1372.0f, 1388.0f, 1404.0f, 1420.0f, 1436.0f, 1452.0f, 
    1468.0f, 1484.0f, 1500.0f, 1516.0f, 1532.0f, 1548.0f, 1564.0f, 1580.0f, 
    1596.0f, 1612.0f, 1628.0f, 1644.0f, 1660.0f, 1676.0f, 1692.0f, 1708.0f, 
    -1708.0f, -1692.0f, -1676.0f, -1660.0f, -1644.0f, -1628.0f, -1612.0f, -1596.0f, 
    -1580.0f, -1564.0f, -1548.0f, -1532.0f, -1516.0f, -1500.0f, -1484.0f, -1468.0f, 
    -1452.0f, -1436.0f, -1420.0f, -1404.0f, -1388.0f, -1372.0f, -1356.0f, -1340.0f, 
    -1324.0f, -1308.0f, -1292.0f, -1276.0f, -1260.0f, -1244.0f, -1228.0f, -1212.0f, 
    -1196.0f, -1180.0f, -1164.0f, -1148.0f, -1132.0f, -1116.0f, -1100.0f, -1084.0f, 
    -1068.0f, -1052.0f, -1036.0f, -1020.0f, -1004.0f, -988.0f, -972.0f, -956.0f, 
    -940.0f, -924.0f, -908.0f, -892.0f, -876.0f, -860.0f, -844.0f, -828.0f, 
    -812.0f, -796.0f, -780.0f, -764.0f, -748.0f, -732.0f, -716.0f, -700.0f, 
    -684.0f, -668.0f, -652.0f, -636.0f, -620.0f, -604.0f, -588.0f, -572.0f, 
    -556.0f, -540.0f, -524.0f, -509.0f, -494.0f, -480.0f, -466.0f, -452.0f, 
    -439.0f, -426.0f, -413.0f, -400.0f, -388.0f, -376.0f, -364.0f, -352.0f, 
    -340.0f, -329.0f, -318.0f, -307.0f, -296.0f, -285.0f, -274.0f, -264.0f, 
    -254.0f, -244.0f, -234.0f, -224.0f, -214.0f, -204.0f, -194.0f, -185.0f, 
    -176.0f, -167.0f, -158.0f, -149.0f, -140.0f, -131.0f, -122.0f, -113.0f, 
    -104.0f, -96.0f, -88.0f, -80.0f, -72.0f, -64.0f, -56.0f, -48.0f, 
    -40.0f, -32.0f, -24.0f, -16.0f, -8.0f, -4.0f, -1.0f, -0.0f
};

template <typename ITER>
static float fpvc_internal_encode(std::vector<uint8_t>& vec, float coeff,
                                  ITER first, ITER last) {
    static_assert(int(0.95f) == 0 && 
                  int(1.05f) == 1 && 
                  int(1.95f) == 1,
                  "float to integer truncation is faulty");
    float mag = 0;
    for (auto it = first; it != last; ++it) {
        const float x = coeff * *it;
        unsigned y;
        if (std::signbit(x)) {  // negative or -0.0f
            y = fpvc_unsigned_compress(int(-x+0.5f));
            if (y > 127) y = 127;
            vec.push_back(uint8_t(255-y));
        }
        else {  // positive or +0.0f
            y = fpvc_unsigned_compress(int(x+0.5f));
            if (y > 127) y = 127;
            vec.push_back(uint8_t(y));
        }
        const auto x2 = float(fpvc_unsigned_decompress(y));
        mag += x2*x2;
    }
    return sqrtf(mag);
}

fpvc_vector_type
internal::fpvc_vector_compress(stdx::forward_iterator<float> first,
                               stdx::forward_iterator<float> last,
                               bool no_opt) {
    std::size_t n = 0;
    float max_val = 0.0f;
    float norm_target = 0;
    for (auto it = first; it != last; ++it, ++n) {
        const float x = fabsf(*it);
        if (max_val < x)
            max_val = x;
        norm_target += *it * *it;
    }
    norm_target = sqrtf(norm_target);
    const auto padded_size = (n+3) & ~std::size_t(3);
    fpvc_vector_type result;
    result.second.reserve(padded_size);
    if (max_val <= 0.0f || norm_target <= 0.0f) {
        result.first = 0.0f;
        result.second.assign(n,0);
    }
    else {
        // note: fpvc_unsigned_compress encoding of 1708 is 127
        result.first = max_val / 1708.0f;
        float coeff = 1.0f / result.first;
        float norm_cur =
            result.first * fpvc_internal_encode(result.second,coeff,first,last);
        float err_cur = fabsf(norm_cur - norm_target)
            / std::min(norm_cur, norm_target);
        FILE_LOG(logTRACE) << "err init: " << err_cur;
        //if (max_init_err < err_cur) max_init_err = err_cur;
        if (err_cur*32 > 1)
            FILE_LOG(logWARNING) << "bad fpvc encoding: error = " << err_cur;
        else if (!no_opt) {
            std::vector<uint8_t> vec;
            vec.reserve(padded_size);
            do {
                vec.clear();
                coeff *= norm_target / norm_cur;
                const float norm_next =
                    result.first * fpvc_internal_encode(vec,coeff,first,last);
                const float err_next = fabsf(norm_next - norm_target)
                    / std::min(norm_next, norm_target);
                if (err_next >= err_cur || result.second == vec)
                    break;  // no improvement in encoding
                FILE_LOG(logTRACE) << "err next: " << err_next;
                result.second.swap(vec);
                norm_cur = norm_next;
                err_cur = err_next;
            } while (true);
        }
        //if (max_final_err < err_cur) max_final_err = err_cur;
    }
    AR_ASSERT(result.second.size() == n);
    return result;
}

std::vector<float>
internal::fpvc_vector_decompress(const fpvc_vector_type& enc) {
    std::vector<float> result;
    result.reserve(enc.second.size());
    fpvc_vector_decompress(back_inserter(result),enc);
    return result;
}

void internal::fpvc_vector_serialize(std::vector<unsigned char>& dest,
                                     const fpvc_vector_type& vec) {
    serialize_value<float>(dest, vec.first);
    const auto n = (vec.second.size()+3) & ~std::size_t(3);
    if (vec.second.capacity() < n)
        throw std::runtime_error("cannot serialize vector without padding");
    dest.insert(dest.end(), vec.second.data(), vec.second.data() + n);
}
        
fpvc_vector_type internal::fpvc_vector_deserialize(
    const void* src, std::size_t vector_size) {
    fpvc_vector_type result;
    result.first = deserialize_value<float>(src);
    if (!(result.first >= 0))
        throw std::domain_error("invalid vector serialization");
    result.second.reserve((vector_size+3) & ~std::size_t(3));
    const uint8_t* p = static_cast<const uint8_t*>(src) + 4;
    result.second.assign(p, p + vector_size);
    return result;
}

fpvc_vector_type internal::fpvc_vector_deserialize(
    std::istream& in, std::size_t vector_size) {
    fpvc_vector_type result;
    result.first = deserialize_value<float>(in);
    if (!(result.first >= 0))
        throw std::domain_error("invalid vector serialization");
    const auto padded_size = (vector_size+3) & ~std::size_t(3);
    result.second.reserve(padded_size);
    result.second.resize(vector_size);
    const auto psi = std::streamsize(padded_size);
    assert(psi >= 0);
    in.read(reinterpret_cast<char*>(result.second.data()),psi);
    if (in.gcount() != psi)
        throw std::runtime_error("read from stream failed");
    return result;
}


/****************  fp16vec methods  ****************/

fp16vec internal::to_fp16vec(const fpvc_vector_type& vec) {
    fp16vec r;
    r.coeff = vec.first;
    r.resize(vec.second.size());
    std::copy_n(
        fpvc_s16_decompress_iterator(vec.second.begin()), r.size(),
        r.begin());
    return r;
}

static constexpr auto ilog2(unsigned n) {
    // this could be replaced with C++20 std::bit_width()
    unsigned l = 0;
    for (auto k : { 8u, 4u, 2u, 1u })
        if (n >= (1u << k)) {
            l += k;
            n >>= k;
        }
    return l;
}
static_assert(0 == ilog2(0));
static_assert(0 == ilog2(1)    && 1 == ilog2(2));
static_assert(1 == ilog2(3)    && 2 == ilog2(4));
static_assert(2 == ilog2(5)    && 2 == ilog2(6));
static_assert(2 == ilog2(7)    && 3 == ilog2(8));
static_assert(3 == ilog2(15)   && 4 == ilog2(16));
static_assert(4 == ilog2(31)   && 5 == ilog2(32));
static_assert(5 == ilog2(63)   && 6 == ilog2(64));
static_assert(6 == ilog2(127)  && 7 == ilog2(128));
static_assert(7 == ilog2(255)  && 8 == ilog2(256));
static_assert(8 == ilog2(511)  && 9 == ilog2(512));
static_assert(9 == ilog2(1023) && 10 == ilog2(1024));

unsigned internal::bits_required(const fp16vec& vec) {
    unsigned n = 0;
    for (int x : vec)
        n = std::max(n, ilog2(unsigned(std::abs(x))));
    return n + 2;
}

void
internal::serialize_8(std::vector<unsigned char>& dest, const fp16vec& vec) {
    int maxabs = 0;
    for (int x : vec)
        maxabs = std::max(maxabs, std::abs(x));

    float coeff = vec.coeff;
    if (!(0 <= coeff))
        throw std::invalid_argument("invalid vector coefficient");
    if (128 <= maxabs)
        coeff *= (float(maxabs) / 127);
    serialize_value<int16_t>(dest, reinterpret_cast<int16_t*>(&coeff)[1]);

    if (maxabs < 128)
        for (auto x : vec)
            dest.push_back(uint8_t(x));
    else {
        const auto r = maxabs / 2; // for rounding
        for (int x : vec) {
            if (0 <= x)
                x = (127*x + r) / maxabs;
            else
                x = (127*x - r) / maxabs;
            dest.push_back(uint8_t(x));
        }
    }
}

void
internal::serialize_12(std::vector<unsigned char>& dest, const fp16vec& vec) {
    auto n = fp16vec_12_bytes(vec.size());
    assert(n >= 4);
    n -= 4;
    serialize_value<float>(dest, vec.coeff);
    std::size_t i = 1;
    for ( ; i < vec.size(); i += 2, n -= 3) {
        assert(n >= 3);
        const auto x0i = vec.values[i-1];
        const auto x1i = vec.values[i];
        if (x0i < -2048 || 2048 <= x0i ||
            x1i < -2048 || 2048 <= x1i)
            throw std::invalid_argument("value out of range for 12-bit");
        const auto x0u = unsigned(x0i) & 0xfffu;
        const auto x1u = unsigned(x1i) & 0xfffu;
        dest.emplace_back(x0u);
        dest.emplace_back((x0u>>8)+(x1u<<4));
        dest.emplace_back(x1u>>4);
    }
    if (i == vec.size()) {
        assert(n >= 2);
        const auto xi = vec.values[i-1];
        if (xi < -2048 || 2048 <= xi)
            throw std::invalid_argument("value out of range for 12-bit");
        dest.emplace_back(xi);
        dest.emplace_back(xi>>8);  // don't care about top 4 bits
        n -= 2;
    }
    dest.insert(dest.end(), n, 0); // padding
}

void
internal::serialize_16(std::vector<unsigned char>& dest, const fp16vec& vec) {
    serialize_value<float>(dest, vec.coeff);
    for (auto x : vec)
        serialize_value<int16_t>(dest, x);
    if (vec.size() & 1)
        dest.insert(dest.end(), 2, 0); // padding
}

fp16vec
internal::deserialize_fp16vec_12(const void* src, std::size_t vector_size) {
    fp16vec result;
    result.coeff = deserialize_value<float>(src);
    if (!(result.coeff >= 0))
        throw std::domain_error("invalid vector serialization");
    result.resize(vector_size);
    auto p = static_cast<const uint8_t*>(src) + 4;
    auto dest = result.begin();
    for ( ; vector_size >= 2; vector_size -= 2, p += 3) {
        *dest++ = int16_t(int16_t((p[1]<<12) + (p[0]<<4)) >> 4);
        *dest++ = int16_t(int16_t((p[2]<<8) + p[1]) >> 4);
    }
    if (vector_size > 0)
        *dest = int16_t(int16_t((p[1]<<12) + (p[0]<<4)) >> 4);
    return result;
}

fp16vec
internal::deserialize_fp16vec_16(const void* src, std::size_t vector_size) {
    fp16vec result;
    result.coeff = deserialize_value<float>(src);
    if (!(result.coeff >= 0))
        throw std::domain_error("invalid vector serialization");
    result.resize(vector_size);
    auto p = static_cast<const int16_t*>(src) + 2;
    auto dest = result.begin();
    for ( ; vector_size > 0; --vector_size, ++p, ++dest)
        *dest = deserialize_value<int16_t>(p);
    return result;
}


/****************  prototype methods  ****************/

stdx::binary internal::serialize(
    unsigned version,
    stdx::forward_iterator<const fpvc_vector_type&> first,
    stdx::forward_iterator<const fpvc_vector_type&> last) {

    unsigned nvecs = 0;
    std::size_t final_len = 4;  // bytes
    for (auto it = first; it != last; ++it) {
        if (!((*it).first > 0))
            throw std::invalid_argument("invalid vector coefficient");
        if ((*it).second.empty())
            throw std::invalid_argument("cannot serialize empty vector");
        final_len += 4 + fpvc_vector_serialize_size(*it);
        ++nvecs;
    }

    if (nvecs <= 0 || 256 <= nvecs)
        throw std::invalid_argument("cannot serialize feature vectors");

    serialize_buffer_type buf;
    buf.reserve(final_len);

    // header
    serialize_value<uint32_t>(buf, version + (0x10<<16) + (nvecs<<24));

    for (auto it = first; it != last; ++it) {
        const auto n = uint32_t((*it).second.size());
        assert(n == (*it).second.size());
        serialize_value<uint32_t>(buf, n);
        fpvc_vector_serialize(buf, *it);
    }

    assert(buf.size() == final_len);
    return buf;
}

stdx::binary internal::serialize(
    unsigned version,
    stdx::forward_iterator<const fp16vec&> first,
    stdx::forward_iterator<const fp16vec&> last,
    unsigned bits_per_element) {

    const bool b12 = bits_per_element <= 12;

    long els_per_vector = 0;
    unsigned nvecs = 0;
    std::size_t final_len = 4;  // bytes
    for (auto it = first; it != last; ++it) {
        if (!((*it).coeff > 0))
            throw std::invalid_argument("invalid vector coefficient");
        const auto n = (*it).size();
        if (n <= 0)
            throw std::invalid_argument("cannot serialize empty vector");
        if (els_per_vector == 0)
            els_per_vector = long(n);
        else if (els_per_vector != long(n))
            els_per_vector = -1;  // not all same length
        final_len += b12 ? fp16vec_12_bytes(n) : fp16vec_16_bytes(n);
        ++nvecs;
    }
    if (nvecs <= 0 || 256 <= nvecs)
        throw std::invalid_argument("cannot serialize feature vectors");

    uint32_t header = version + ((b12?0x11:0x12)<<16) + (nvecs<<24);

    const bool same_size = 0 < els_per_vector && els_per_vector < 256;
    if (same_size)
        header += unsigned(els_per_vector) << 8;
    else
        final_len += 4*nvecs;

    serialize_buffer_type buf;
    buf.reserve(final_len);
    serialize_value<uint32_t>(buf, header);

    for (auto it = first; it != last; ++it) {
        if (!same_size) {
            const auto n = uint32_t((*it).size());
            assert(n == (*it).size());
            serialize_value<uint32_t>(buf, n);
        }
        if (b12) serialize_12(buf, *it);
        else serialize_16(buf, *it);
    }

    assert(buf.size() == final_len);
    return buf;
}

std::vector<std::pair<fp16vec, fpvc_vector_type> >
internal::deserialize_fpvc(const void* src, std::size_t len) {
    if (len < 4)
        throw std::domain_error("invalid prototype serialization (too short)");

    const auto u8 = static_cast<const uint8_t*>(src);
    // note: u8[0] is prototype version number
    const unsigned nels_head = u8[1];

    if (nels_head == 1 && 132 <= len) {
        // single vector of 128 int8 values and bfloat16 coeff
        std::vector<std::pair<fp16vec, fpvc_vector_type> > result(1);
        auto& v = result.front().first;
        v.resize(128);
        std::copy_n(static_cast<const int8_t*>(src)+4, 128, v.begin());
        const auto ci = uint32_t(deserialize_value<uint16_t>(u8+2)) << 16;
        memcpy(&v.coeff, &ci, sizeof(float));
        if (!(0 <= v.coeff))
            throw std::domain_error("invalid vector serialization (bad coeff)");
        return result;
    }

    const unsigned type = u8[2];
    const unsigned nvecs = u8[3];
    if (nvecs == 0)
        throw std::domain_error("invalid prototype serialization (empty)");

    auto u32 = static_cast<const uint32_t*>(src) + 1;
    len -= sizeof(uint32_t);

    std::vector<std::pair<fp16vec, fpvc_vector_type> > result(nvecs);
    for (auto& vec : result) {
        auto nels = nels_head;
        if (nels == 0) {
            if (len < sizeof(uint32_t))
                throw std::domain_error(
                    "invalid prototype serialization (too short)");
            nels = deserialize_value<uint32_t>(u32);
            if (nels == 0)
                throw std::domain_error(
                    "invalid prototype serialization (empty vector)");
            ++u32;
            len -= sizeof(uint32_t);
        }

        std::size_t n;
        switch (type) {
        case 0x10:
            n = fpvc_vector_serialize_size(nels);
            if (len < n)
                throw std::domain_error("invalid prototype serialization (too short)");
            vec.second = fpvc_vector_deserialize(u32,nels);
            vec.first = to_fp16vec(vec.second);
            break;

        case 0x11:
            n = fp16vec_12_bytes(nels);
            if (len < n)
                throw std::domain_error("invalid prototype serialization (too short)");
            vec.first = deserialize_fp16vec_12(u32,nels);
            break;

        case 0x12:
            n = fp16vec_16_bytes(nels);
            if (len < n)
                throw std::domain_error("invalid prototype serialization (too short)");
            vec.first = deserialize_fp16vec_16(u32,nels);
            break;

        default:
            throw std::domain_error("invalid prototype serialization (format)");
        }
        assert((n&3) == 0);
        u32 += n/4;
        len -= n;
        assert(vec.first.size() == nels);
    }
    if (len > 0)
        FILE_LOG(logWARNING) << "prototype serialization has extra bytes: "
                             << len;
    return result;
}
