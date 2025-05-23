
#include "io.hpp"

#include <cstring>
#include <cstddef>
#include <functional>
#include <ostream>

#include <applog/core.hpp>

using namespace raw_image;

namespace {
    class point3f_at {
        const std::byte* m_base;
        unsigned m_stride;
        unsigned m_size;

    public:
        template <typename T>
        point3f_at(const T* first, unsigned size)
            : m_base(reinterpret_cast<const std::byte*>(
                         static_cast<const point3f*>(first))),
              m_stride(sizeof(T)),
              m_size(size) {
        }

        template <typename T>
        point3f_at(const stdx::span<T>& s)
            : point3f_at(s.data(), unsigned(s.size())) {
        }

        inline const point3f& operator[](unsigned idx) const {
            return *reinterpret_cast<const point3f*>(m_base + idx*m_stride);
        }
        const point3f& at(unsigned idx) const {
            if (idx >= m_size)
                std::out_of_range("vertex index out of range");
            return operator[](idx);
        }
    };
}

static void save_stl_impl(
    point3f_at verts,
    stdx::forward_iterator<const index_list&> face_first,
    stdx::forward_iterator<const index_list&> face_last,
    std::string_view comment, std::ostream& out) {

    // header
    std::string desc(comment);
    desc.resize(79,' ');
    out.write(desc.c_str(), 80);

    // number of triangles
    const auto ntri = uint32_t(std::distance(face_first, face_last));
    out.write(reinterpret_cast<const char*>(&ntri), 4);

    struct triangle_record {
        point3f n, v0, v1, v2;
        float extra;
    };
    triangle_record tri;
    static_assert(sizeof(tri) == 13*sizeof(float));
    memset(&tri, 0, sizeof(tri));

    for ( ; face_first != face_last; ++face_first) {
        if (!face_first->is_triangle())
            throw std::invalid_argument("stl files only support triangles");
        auto& v0 = tri.v0 = verts[face_first->indices[0]];
        auto& v1 = tri.v1 = verts[face_first->indices[1]];
        auto& v2 = tri.v2 = verts[face_first->indices[2]];
        tri.n = cross(v0 - v1, v1 - v2);
        tri.n /= std::sqrt(dot(tri.n,tri.n));
        out.write(reinterpret_cast<const char*>(&tri), 50);
    }
}

void raw_image::save_stl(
    stdx::span<const point3f> verts,
    stdx::forward_iterator<const index_list&> face_first,
    stdx::forward_iterator<const index_list&> face_last,
    std::string_view comment, std::ostream& out) {
    save_stl_impl(verts, face_first, face_last, comment, out);
}

void raw_image::save_stl(
    stdx::span<const point3f_rgbf> verts,
    stdx::forward_iterator<const index_list&> face_first,
    stdx::forward_iterator<const index_list&> face_last,
    std::string_view comment, std::ostream& out) {
    save_stl_impl(verts, face_first, face_last, comment, out);
}


static void write(float x, std::ostream& out) {
    out.write(reinterpret_cast<const char*>(&x), sizeof(x));
}
static void write(unsigned x, std::ostream& out) {
    out.write(reinterpret_cast<const char*>(&x), sizeof(x));
}

void raw_image::save_ply(
    stdx::span<const point3f_rgbf> verts,
    stdx::forward_iterator<const index_list&> face_first,
    stdx::forward_iterator<const index_list&> face_last,
    std::string_view comment, std::ostream& out) {

    const auto nfaces = std::distance(face_first, face_last);
    if (nfaces <= 0 || verts.empty())
        throw std::invalid_argument("ply model is empty");

    for (auto c : comment)
        if (c < ' ' || 127 <= c)
            throw std::invalid_argument("invalid ply comment");

    out << "ply" << std::endl
        //<< "format ascii 1.0" << std::endl
        << "format binary_little_endian 1.0" << std::endl;
    if (!comment.empty())
        out << "comment " << comment << std::endl;
    out << "element vertex " << verts.size() << std::endl
        << "property float x" << std::endl
        << "property float y" << std::endl
        << "property float z" << std::endl
        << "property uchar red" << std::endl
        << "property uchar green" << std::endl
        << "property uchar blue" << std::endl
        << "element face " << nfaces << std::endl
        << "property list uchar int vertex_indices" << std::endl
        << "end_header" << std::endl;

    /* if (0) { // ascii
        for (auto& v : vertices)
            out << v.xyz[0] << ' ' << -v.xyz[1] << ' '
                << int(min_depth) - v.xyz[2] << ' '
                << unsigned(v.rgb[0]) << ' '
                << unsigned(v.rgb[1]) << ' '
                << unsigned(v.rgb[2]) << std::endl;
        for (auto& t : triangles)
            out << 3 << ' ' << t[2] << ' ' << t[1]
                << ' ' << t[0] << std::endl;
        for (auto& q : squares)
            out << 4 << ' ' << q[3] << ' ' << q[2] << ' ' << q[1]
                << ' ' << q[0] << std::endl;
       }
    */

    // binary
    for (auto& v : verts) {
        write(v.x, out);
        write(v.y, out);
        write(v.z, out);
        out.write(reinterpret_cast<const char*>(&v.r), 3);
    }
    for ( ; face_first != face_last; ++face_first) {
        if (face_first->is_triangle()) {
            const char c3 = 3;
            out.write(&c3,1);
            write(face_first->indices[2],out);
            write(face_first->indices[1],out);
            write(face_first->indices[0],out);
        }
        else if (face_first->is_quad()) {
            const char c4 = 4;
            out.write(&c4,1);
            write(face_first->indices[3],out);
            write(face_first->indices[2],out);
            write(face_first->indices[1],out);
            write(face_first->indices[0],out);
        }
        else throw std::invalid_argument(
            "polygon face must be triangle or quadrilateral");
    }
}

void raw_image::save_ply(
    stdx::span<const point3f> verts,
    stdx::forward_iterator<const indices_rgbf&> face_first,
    stdx::forward_iterator<const indices_rgbf&> face_last,
    std::string_view comment, std::ostream& out) {

    const auto nfaces = std::distance(face_first, face_last);
    if (nfaces <= 0 || verts.empty())
        throw std::invalid_argument("ply model is empty");

    for (auto c : comment)
        if (c < ' ' || 127 <= c)
            throw std::invalid_argument("invalid ply comment");

    out << "ply" << std::endl
        //<< "format ascii 1.0" << std::endl
        << "format binary_little_endian 1.0" << std::endl;
    if (!comment.empty())
        out << "comment " << comment << std::endl;
    out << "element vertex " << verts.size() << std::endl
        << "property float x" << std::endl
        << "property float y" << std::endl
        << "property float z" << std::endl
        << "element face " << nfaces << std::endl
        << "property list uchar int vertex_indices" << std::endl
        << "property uchar red" << std::endl
        << "property uchar green" << std::endl
        << "property uchar blue" << std::endl
        << "end_header" << std::endl;

    /* if (0) { // ascii
        for (auto& v : vertices)
            out << v.xyz[0] << ' ' << -v.xyz[1] << ' '
                << int(min_depth) - v.xyz[2] << ' '
                << unsigned(v.rgb[0]) << ' '
                << unsigned(v.rgb[1]) << ' '
                << unsigned(v.rgb[2]) << std::endl;
        for (auto& t : triangles)
            out << 3 << ' ' << t[2] << ' ' << t[1]
                << ' ' << t[0] << std::endl;
        for (auto& q : squares)
            out << 4 << ' ' << q[3] << ' ' << q[2] << ' ' << q[1]
                << ' ' << q[0] << std::endl;
       }
    */

    // binary
    for (auto& v : verts) {
        write(v.x, out);
        write(v.y, out);
        write(v.z, out);
    }
    for ( ; face_first != face_last; ++face_first) {
        if (face_first->is_triangle()) {
            const char c3 = 3;
            out.write(&c3,1);
            write(face_first->indices[2],out);
            write(face_first->indices[1],out);
            write(face_first->indices[0],out);
        }
        else if (face_first->is_quad()) {
            const char c4 = 4;
            out.write(&c4,1);
            write(face_first->indices[3],out);
            write(face_first->indices[2],out);
            write(face_first->indices[1],out);
            write(face_first->indices[0],out);
        }
        else throw std::invalid_argument(
            "polygon face must be triangle or quadrilateral");
        out.write(reinterpret_cast<const char*>(&face_first->r), 3);
    }
}

template <typename T, typename DEST>
static std::size_t
prop_discard(const uint8_t*, std::size_t size, DEST&) {
    if (size < sizeof(T))
        throw std::runtime_error("unexpected end of ply file");
    return sizeof(T);
}

template <typename T, typename SUB, T SUB::*M, typename DEST>
static std::size_t
prop_single(const uint8_t* src, std::size_t size, DEST& dest) {
    if (size < sizeof(T))
        throw std::runtime_error("unexpected end of ply file");
    static_cast<SUB&>(dest).*M = *reinterpret_cast<const T*>(src);
    return sizeof(T);
}

template <typename From, typename To, typename SUB, To SUB::*M, typename DEST>
static std::size_t
prop_conv(const uint8_t* src, std::size_t size, DEST& dest) {
    if (size < sizeof(From))
        throw std::runtime_error("unexpected end of ply file");
    static_cast<SUB&>(dest).*M = To(*reinterpret_cast<const From*>(src));
    return sizeof(From);
}

template <typename T, typename DEST>
static std::size_t
prop_list_discard(const uint8_t* src, std::size_t size, DEST&) {
    if (size <= 0 || size <= sizeof(T)**src)
        throw std::runtime_error("unexpected end of ply file");
    return 1 + sizeof(T)**src;
}

template <typename T, typename DEST>
static std::size_t
prop_list_indices(const uint8_t* src, std::size_t size, DEST& dest) {
    if (size <= 0 || size <= sizeof(T)**src)
        throw std::runtime_error("unexpected end of ply file");
    const auto n = unsigned(*src++);
    auto* ptr = reinterpret_cast<const T*>(src);
    auto& list = static_cast<index_list&>(dest);
    switch (n) {
    case 3: list = { ptr[0], ptr[1], ptr[2] }; break;
    case 4: list = { ptr[0], ptr[1], ptr[2], ptr[3] }; break;
    default:
        throw std::runtime_error("only triangle and quad faces are supported");
    }
    return 1 + n*sizeof(T);
}

std::tuple<std::vector<point3f_rgbf_uv>,
           std::vector<indices_rgbf>,
           ply_options,
           std::vector<std::string> >
raw_image::load_ply(const void* vdata, std::size_t size) {
    auto* data = static_cast<const uint8_t*>(vdata);
    std::vector<std::string_view> header;
    for (;;) {
        unsigned len = 0;
        while (len < size && ' ' <= data[len] && data[len] < 127)
            ++len;
        if (!(len < size && data[len] == '\n'))
            throw std::runtime_error("ply file has malformed header");
        auto& s = header.emplace_back(reinterpret_cast<const char*>(data), len);
        data += ++len;
        size -= len;
        if (s == "end_header") {
            header.pop_back();
            break;
        }
    }
    if (header.empty() || header.front() != "ply")
        throw std::runtime_error("ply file does not begin with 'ply'");
    header.erase(header.begin());

    std::tuple<std::vector<point3f_rgbf_uv>,
               std::vector<indices_rgbf>,
               ply_options,
               std::vector<std::string> > t;
    auto& verts = std::get<0>(t);
    auto& faces = std::get<1>(t);
    auto& comment = std::get<3>(t);
    unsigned flag = 0;

    std::string_view format;
    unsigned reading_element = 0; // 1 = vertices, 2 = faces

    std::vector<std::size_t(*)(const uint8_t*, std::size_t, point3f_rgbf_uv&)> vert_props;
    std::vector<std::size_t(*)(const uint8_t*, std::size_t, indices_rgbf&)> face_props;

    std::vector<std::string_view> fields;
    for (const auto& line : header) {
        fields.clear();
        for (auto s = line; !s.empty(); ) {
            const auto delim = s.find(' ');
            if (delim < s.size()) {
                fields.emplace_back(s.substr(0,delim));
                s.remove_prefix(delim+1);
            }
            else {
                fields.emplace_back(s);
                break;
            }
        }
        if (fields.empty()) {
            FILE_LOG(logWARNING) << "ply header has blank line";
            continue;
        }

        if (fields.front() == "format") {
            if (fields.size() <= 1)
                throw std::runtime_error("ply file has invalid format");
            if (!format.empty())
                throw std::runtime_error("ply file specifies multiple formats");
            const auto len =
                fields.back().data() + fields.back().size() - fields[1].data();
            format = std::string_view(fields[1].data(), std::size_t(len));
        }

        else if (fields.front() == "comment") {
            if (1 < fields.size())
                comment.emplace_back(
                    fields[1].data(),
                    fields.back().data() + fields.back().size()
                    );
        }

        else if (fields.front() == "element") {
            if (fields.size() != 3)
                throw std::runtime_error("ply element has invalid format");
            const auto ns = std::string(fields[2]);
            const auto nl = atol(ns.c_str());
            const auto nu = std::size_t(nl);
            if (nl <= 0 || size <= nu || ns != std::to_string(nu))
                throw std::runtime_error("ply element has invalid count");
            if (fields[1] == "vertex") {
                if (!verts.empty())
                    throw std::runtime_error(
                        "ply element vertex specified twice");
                verts.resize(nu);
                reading_element = 1;
            }
            else if (fields[1] == "face") {
                if (!faces.empty())
                    throw std::runtime_error(
                        "ply element face specified twice");
                faces.resize(nu);
                reading_element = 2;
            }
            else {
                FILE_LOG(logERROR) << "ply element '" << fields[1]
                                   << "' unsupported";
                throw std::runtime_error("ply element unsupported");
            }
        }

        else if (fields.front() == "property") {
            if (fields.size() < 3)
                throw std::runtime_error("ply property has invalid format");

            if (reading_element == 1) { // vertex
                if (fields[1] == "float") {
                    if (fields[2] == "x")
                        vert_props.push_back(
                            &prop_single<float, point3f, &point3f::x>);
                    else if (fields[2] == "y")
                        vert_props.push_back(
                            &prop_single<float, point3f, &point3f::y>);
                    else if (fields[2] == "z")
                        vert_props.push_back(
                            &prop_single<float, point3f, &point3f::z>);
                    else if (fields[2] == "texture_u") {
                        vert_props.push_back(&prop_single<float, point3f_rgbf_uv, &point3f_rgbf_uv::u>);
                        flag |= ply_vertex_has_uv;
                    }
                    else if (fields[2] == "texture_v") {
                        vert_props.push_back(&prop_single<float, point3f_rgbf_uv, &point3f_rgbf_uv::v>);
                        flag |= ply_vertex_has_uv;
                    }
                    else {
                        FILE_LOG(logWARNING) << "ignoring vertex " << line;
                        vert_props.push_back(&prop_discard<float>);
                    }
                }
                else if (fields[1] == "double") {
                    if (fields[2] == "x")
                        vert_props.push_back(
                            &prop_conv<double, float, point3f, &point3f::x>);
                    else if (fields[2] == "y")
                        vert_props.push_back(
                            &prop_conv<double, float, point3f, &point3f::y>);
                    else if (fields[2] == "z")
                        vert_props.push_back(
                            &prop_conv<double, float, point3f, &point3f::z>);
                    else if (fields[2] == "texture_u") {
                        vert_props.push_back(&prop_conv<double, float, point3f_rgbf_uv, &point3f_rgbf_uv::u>);
                        flag |= ply_vertex_has_uv;
                    }
                    else if (fields[2] == "texture_v") {
                        vert_props.push_back(&prop_conv<double, float, point3f_rgbf_uv, &point3f_rgbf_uv::v>);
                        flag |= ply_vertex_has_uv;
                    }
                    else {
                        FILE_LOG(logWARNING) << "ignoring vertex " << line;
                        vert_props.push_back(&prop_discard<double>);
                    }
                }
                else if (fields[1] == "uchar") {
                    if (fields[2] == "red") {
                        vert_props.push_back(
                            &prop_single<uint8_t, rgbf, &rgbf::r>);
                        flag |= ply_vertex_has_rgb;
                    }
                    else if (fields[2] == "green") {
                        vert_props.push_back(
                            &prop_single<uint8_t, rgbf, &rgbf::g>);
                        flag |= ply_vertex_has_rgb;
                    }
                    else if (fields[2] == "blue") {
                        vert_props.push_back(
                            &prop_single<uint8_t, rgbf, &rgbf::b>);
                        flag |= ply_vertex_has_rgb;
                    }
                    else if (fields[2] == "alpha")
                        vert_props.push_back(
                            &prop_single<uint8_t, rgbf, &rgbf::flag>);
                    else {
                        FILE_LOG(logWARNING) << "ignoring vertex " << line;
                        vert_props.push_back(&prop_discard<uint8_t>);
                    }
                }
                else
                    throw std::runtime_error(
                        "ply vertex property has unsupported format");
            }

            else if (reading_element == 2) { // face
                if (fields[1] == "uchar") {
                    if (fields[2] == "red") {
                        face_props.push_back(
                            &prop_single<uint8_t, rgbf, &rgbf::r>);
                        flag |= ply_face_has_rgb;
                    }
                    else if (fields[2] == "green") {
                        face_props.push_back(
                            &prop_single<uint8_t, rgbf, &rgbf::g>);
                        flag |= ply_face_has_rgb;
                    }
                    else if (fields[2] == "blue") {
                        face_props.push_back(
                            &prop_single<uint8_t, rgbf, &rgbf::b>);
                        flag |= ply_face_has_rgb;
                    }
                    else if (fields[2] == "alpha")
                        face_props.push_back(
                            &prop_single<uint8_t, rgbf, &rgbf::flag>);
                    else {
                        FILE_LOG(logWARNING) << "ignoring face " << line;
                        face_props.push_back(&prop_discard<uint8_t>);
                    }
                }
                else if (fields[1] == "list") {
                    if (fields.size() != 5 || fields[2] != "uchar")
                        throw std::runtime_error("ply list has invalid format");
                    if (fields[3] == "int") {
                        if (fields[4] == "vertex_indices")
                            face_props.push_back(&prop_list_indices<uint32_t>);
                        else {
                            FILE_LOG(logWARNING) << "ignoring face " << line;
                            face_props.push_back(&prop_list_discard<int32_t>);
                        }
                    }
                    else if (fields[3] == "float") {
                        FILE_LOG(logWARNING) << "ignoring face " << line;
                        face_props.push_back(&prop_list_discard<float>);
                    }
                    else
                        throw std::runtime_error("ply list has invalid format");
                }
                else
                    throw std::runtime_error(
                        "ply face property has unsupported format");
            }
            else
                throw std::runtime_error("ply property without element");
        }

        else
            FILE_LOG(logWARNING) << "ignoring ply '" << line << '\'';
    }

    if (format != "binary_little_endian 1.0") {
        FILE_LOG(logERROR) << "support for ply format '" << format
                           << "' not implemented";
        throw std::runtime_error("ply file has unsupported format");
    }

    std::get<ply_options>(t) = ply_options(flag);

    // read verts
    for (auto& v : verts)
        for (auto* p : vert_props) {
            auto n = (*p)(data, size, v);
            data += n, size -= n;
        }

    // read faces
    for (auto& v : faces)
        for (auto* p : face_props) {
            auto n = (*p)(data, size, v);
            data += n, size -= n;
        }

    if (0 < size)
        FILE_LOG(logWARNING) << "ply file has " << size << " extra bytes";

    return t;
}
