
#include "tiff.hpp"
#include <raw_image/core.hpp>

#include <tiff.h>
#include <tiffio.h>

#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <applog/core.hpp>


using namespace raw_image;


static std::atomic_flag tiffHandlersSet = ATOMIC_FLAG_INIT;

static auto to_string(const char* fmt, va_list ap) {
    char buf[128];
    if (fmt) {
        vsnprintf(buf, sizeof(buf), fmt, ap);
        buf[sizeof(buf)-1] = 0;
    }
    else buf[0] = 0;
    return std::string(buf);
}

static void tiffErrorHandler(
    const char* module, const char* fmt, va_list ap) {
    FILE_LOG(logERROR) << "TIFF: [" << module << "] " << to_string(fmt,ap);
}

static void tiffWarningHandler(
    const char* module, const char* fmt, va_list ap) {
    FILE_LOG(logWARNING) << "TIFF: [" << module << "] " << to_string(fmt,ap);
}

static void set_handles() {
    if (!tiffHandlersSet.test_and_set()) {
        TIFFSetErrorHandler(&tiffErrorHandler);
        TIFFSetWarningHandler(&tiffWarningHandler);
    }
}

static inline auto get_fileno(FILE* file) {
#ifdef __WIN32__
    return _fileno(file);
#else
    return fileno(file);
#endif
}

namespace {
    struct tiff_close_fn {
        inline void operator()(TIFF* t) const { TIFFClose(t); }
    };
    using tiff_ptr = std::unique_ptr<TIFF,tiff_close_fn>;

    struct data_client {
        const unsigned char* m_data;
        const unsigned char* m_cur;
        std::size_t m_size;

        data_client(const void* data = nullptr, std::size_t size = 0)
            : m_data(static_cast<const unsigned char*>(data)),
              m_cur(m_data),
              m_size(size) {
        }
        
        tsize_t read(tdata_t buffer, tsize_t size) {
            if (size <= 0)
                return 0;
            const auto rem = m_data + m_size - m_cur;
            if (rem <= 0)
                return 0;
            if (size > rem)
                size = rem;
            memcpy(buffer,m_cur,std::size_t(size));
            m_cur += size;
            return size;
        }
        static tsize_t read(thandle_t st, tdata_t buffer, tsize_t size) {
            return static_cast<data_client*>(st)->read(buffer,size);
        }
        
        static tsize_t write(thandle_t, tdata_t, tsize_t) {
            return 0;
        }
        
        static int close(thandle_t) {
            return 0;
        }
        
        toff_t seek(toff_t pos, int whence) {
            if (pos == 0xFFFFFFFF)
                return toff_t(-1);
            const std::size_t upos = pos;
            switch (whence) {
            case SEEK_SET:
                if (upos < m_size)
                    m_cur = m_data + upos;
                else
                    m_cur = m_data + m_size;
                break;
            case SEEK_CUR:
                if (upos < m_size && m_cur + upos < m_data + m_size)
                    m_cur += upos;
                else
                    m_cur = m_data + m_size;
                break;
            case SEEK_END:
                m_cur = m_data + m_size;
                break;
            default:
                return toff_t(-1);
            }
            return toff_t(m_cur - m_data);
        }
        static toff_t seek(thandle_t st, toff_t pos, int whence) {
            return static_cast<data_client*>(st)->seek(pos,whence);
        }
        
        static toff_t size(thandle_t st) {
            return static_cast<data_client*>(st)->m_size;
        }
        
        static int map(thandle_t, tdata_t*, toff_t*) {
            return 0;
        }
        
        static void unmap(thandle_t, tdata_t, toff_t) {
            return;
        }

        TIFF* open(const char* mode) {
            return TIFFClientOpen(
                "memory", mode, this,
                &data_client::read, 
                &data_client::write, 
                &data_client::seek, 
                &data_client::close, 
                &data_client::size,
                &data_client::map, 
                &data_client::unmap);
        }
    };
}

static plane_ptr load(TIFF* tiff) {
    uint32_t width, height;
    TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);

    static constexpr auto ofs = plane_struct_padded_size;
    const auto bytes_per_line = 4 * width;
    const auto nbytes = height * std::size_t(bytes_per_line) + ofs;
    const auto buf = operator new(nbytes);
    std::fill_n(static_cast<unsigned char*>(buf), ofs, 0);

    plane_ptr image(static_cast<plane*>(buf));
    image->data = static_cast<unsigned char*>(buf) + ofs;
    image->width = width;
    image->height = height;
    image->bytes_per_line = bytes_per_line;
    image->layout = pixel::rgba32;

    if (TIFFReadRGBAImageOriented(tiff, width, height,
                                  reinterpret_cast<uint32_t*>(image->data),
                                  ORIENTATION_TOPLEFT) != 1) {
        FILE_LOG(logERROR) << "tiff: failed to decode image";
        throw std::runtime_error("failed to decode tiff image");
    }
    return image;
}

plane_ptr raw_image::tiff_load(FILE* file) {
    set_handles();
    fseek(file, 0, SEEK_SET);
    const auto tiff =
        tiff_ptr(TIFFFdOpen(get_fileno(file), "image.tiff", "rb"));
    if (!tiff)
        throw std::runtime_error("failed to open tiff image");
    return load(tiff.get());
}

plane_ptr raw_image::tiff_load(const void* src, std::size_t size) {
    set_handles();
    data_client data(src,size);
    const auto tiff = tiff_ptr(data.open("r"));
    if (!tiff)
        throw std::runtime_error("failed to access tiff in memory");
    return load(tiff.get());
}
