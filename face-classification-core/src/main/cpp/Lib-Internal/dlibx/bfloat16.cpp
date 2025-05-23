
#include "bfloat16.hpp"
#include "tensor.hpp"

#include <cstdint>
#include <cmath>

using namespace dlibx;

static int manip_parameter_format = -1;

std::ostream& dlibx::set_parameter_format(std::ostream& s, parameter_format f) {
    if (manip_parameter_format < 0)
        manip_parameter_format = std::ios_base::xalloc();
    s.iword(manip_parameter_format) = int(f);
    return s;
}
parameter_format dlibx::get_parameter_format(std::ostream& s) {
    if (manip_parameter_format < 0)
        return parameter_format::native;
    return parameter_format(s.iword(manip_parameter_format));
}

static_assert(sizeof(float) == 4 &&
              sizeof(uint32_t) == 4 &&
              sizeof(uint16_t) == 2);

static const unsigned char bits_table[32] = {
     1, 10,  2, 11, 14, 22,  3, 30,
    12, 15, 17, 19, 23, 26,  4, 31,
     9, 13, 21, 29, 16, 18, 25,  8,
    20, 28, 24,  7, 27,  6,  5, 32,
};
unsigned dlibx::bits_required(unsigned x) {
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return bits_table[uint32_t(x*0x07C4ACDD) >> 27];
}

float* dlibx::truncate_to_bfloat16(float* data, std::size_t size) {
    for ( ; size > 0; --size, ++data) {
        auto u = reinterpret_cast<uint32_t*>(data);
        if (std::isnan(*data))
            *u |= 0x00ff0000u;
        *u &= 0xffff0000u;
    }
    return data;
}

static const dlib::byte_orderer bo;

void bfloat16_const_span::serialize(std::ostream& out) const {
    auto sbuf = out.rdbuf();
    for (float d : *this) {
        const auto nan = std::isnan(d);
        bo.host_to_little(d);
        auto p = reinterpret_cast<char*>(&d) + 2;
        if (nan) *p = char(0xff); // make sure we don't change nan -> inf
        sbuf->sputn(p, 2);
    }
}

void bfloat16_span::deserialize(std::istream& in) {
    // read data into second half of buffer
    auto sbuf = in.rdbuf();
    auto buf = reinterpret_cast<char*>(data()) + 2*size();
    if (sbuf->sgetn(buf, long(2*size())) != long(2*size())) {
        in.setstate(std::ios::badbit);
        throw dlib::serialization_error("Error reading data while deserializing dlib::resizable_tensor.");
    }
    // convert values in order starting at front of buffer
    auto src = reinterpret_cast<uint16_t*>(buf);
    for (auto dest = reinterpret_cast<uint32_t*>(data()),
             end = dest + size(); dest != end; ++dest, ++src) {
        auto x = *src;
        bo.little_to_host(x);
        *dest = uint32_t(x) << 16;
    }
}

void
dlibx::serialize_bfloat16(const dlib::tensor& item, std::ostream& out) {
    using dlib::serialize;
    const int version = -16;
    serialize(version, out);
    serialize(item.num_samples(), out);
    serialize(item.k(), out);
    serialize(item.nr(), out);
    serialize(item.nc(), out);
    serialize(bfloat16(item.host(),item.size()), out);
}

void dlibx::deserialize(dlib::resizable_tensor& item, std::istream& in) {
    using dlib::deserialize;
    if (in.peek() != 0x81)
        dlib::deserialize(item, in);
    else {
        int version;
        deserialize(version, in);
        if (version != -16)
            throw dlib::serialization_error("Unexpected version found while deserializing dlib::resizable_tensor.");
        long long num_samples = 0, k = 0, nr = 0, nc = 0;
        deserialize(num_samples, in);
        deserialize(k, in);
        deserialize(nr, in);
        deserialize(nc, in);
        item.set_size(num_samples, k, nr, nc);
        deserialize(bfloat16(item.host_write_only(), item.size()), in);
    }
}

bool dlibx::is_bfloat16(float const* d, std::size_t n) {
    for (auto u = reinterpret_cast<uint32_t const*>(d); n > 0; --n, ++u)
        if (*u & 0x0ffff)
            return false;
    return true;
}
bool dlibx::is_bfloat16(const dlib::tensor& src) {
    return is_bfloat16(src.host(), src.size());
}
