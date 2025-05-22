#include <boost/test/unit_test.hpp>
#include <applog/core.hpp>
#include <iomanip>

/*
namespace {
    struct bits_set_iterator {
        std::array<unsigned,8> bits = {0};

        explicit bits_set_iterator(unsigned n) {
            auto bit = 128u;
            while (n > 0) {
                bits[--n] = bit;
                bit >>= 1;
            }
        }

        auto operator*() const {
            unsigned x = 0;
            for (auto bit : bits)
                if (bit) x ^= bit; else break;
            return x;
        }

        bool operator!=(const bits_set_iterator& other) const {
            return bits[0] != other.bits[0];
        }

        auto& operator++() {
            unsigned i = 0;
            for (;;) {
                if ((bits[i] >>= 1) != 0)
                    break;
                if (++i == 8) return *this;
            }
            for ( ; i > 0; --i)
                bits[i-1] = bits[i] >> 1;
            return *this;
        }
    };
    struct vals_with_bits_set {
        const unsigned nbits;
        auto begin() const { return bits_set_iterator(nbits); }
        auto end()   const { return bits_set_iterator(0); }
    };
}
*/

/** \brief Popcount (number of bits set) for all byte values.
 */
static constexpr auto bytepop =
    [](){
        std::array<uint8_t,256> x = {};
        for (unsigned i = 0; i < 256; ++i) {
            auto& y = x[i];
            for (auto j = i; j; j >>= 1)
                if (j&1) ++y;
        }
        return x;
    }();


BOOST_AUTO_TEST_SUITE(rec)



BOOST_AUTO_TEST_CASE(bits_set) {
    for (unsigned n = 0; n <= 8; ++n) {
        std::stringstream ss;
        ss << std::setfill('0') << std::hex;
        unsigned count = 0;
        for (unsigned i = 0; i < 256; ++i)
            if (bytepop[i] == n) {
                ss << ' ' << std::setw(2) << i;
                ++count;
            }
        FILE_LOG(logINFO) << n << ' ' << count << '\t' << ss.str();
    }
}

BOOST_AUTO_TEST_SUITE_END()
