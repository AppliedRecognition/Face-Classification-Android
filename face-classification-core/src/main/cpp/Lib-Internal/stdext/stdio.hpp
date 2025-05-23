#pragma once

#include <cstdio>
#include <codecvt>
#include <locale>
#include <memory>
#include <type_traits>


// note: on Windows _wfopen(const wchar_t*, const wchar_t*)
// is available in the global namespace

namespace wfopen_rb {
    template <typename CHAR>
    inline std::enable_if_t<std::is_same_v<CHAR,char>,FILE*>
    _wfopen(const CHAR* filename, const wchar_t*) {
        return fopen(filename,"rb");
    }
    // this method is used by wfopen_tester() below: it is never called
    template <typename CHAR>
    constexpr std::enable_if_t<!std::is_same_v<CHAR,char>,void const*>
    _wfopen(const CHAR*, const wchar_t*) { return nullptr; }
}
namespace wfopen_wb {
    inline FILE* _wfopen(const char* filename, const wchar_t*) {
        return fopen(filename,"wb");
    }
}


namespace stdx {

    /** \brief Get C-style null-terminated string from path object.
     *
     * Either calls the object's c_str() method, or
     * returns const char* or const wchar_t* directly.
     *
     * Acceptable path objects include std::string, std::wstring,
     * std::filesystem::path and boost::filesystem::path.
     */
    constexpr char const* c_str(char* s) { return s; }
    constexpr char const* c_str(char const* s) { return s; }
    constexpr wchar_t const* c_str(wchar_t* s) { return s; }
    constexpr wchar_t const* c_str(wchar_t const* s) { return s; }
    template <typename T>
    inline auto const* c_str(T const& path) { return path.c_str(); }


    /** \brief Generic std::string from path object.
     *
     * Acceptable path objects include std::string, std::wstring,
     * std::filesystem::path and boost::filesystem::path.
     * Also, const char* and const wchar_t*.
     * Convertion to utf8 is done if necessary.
     */
    inline auto generic_string(char const* path) {
        return std::string(path);
    }
    constexpr const auto& generic_string(const std::string& path) {
        return path;
    }
    inline auto generic_string(wchar_t const* path) {
        using convert_type = std::codecvt_utf8<wchar_t>;
        std::wstring_convert<convert_type, wchar_t> converter;
        return converter.to_bytes(path);
    }
    inline auto generic_string(const std::wstring& path) {
        using convert_type = std::codecvt_utf8<wchar_t>;
        std::wstring_convert<convert_type, wchar_t> converter;
        return converter.to_bytes(path);
    }
    template <typename PATH>
    inline std::enable_if_t<std::is_convertible_v<decltype(std::declval<const PATH&>().generic_string()),std::string>,std::string>
    generic_string(const PATH& path) {
        return path.generic_string();
    }


    /** \brief Method to call fclose() used by file_ptr below.
     */
    struct fclose_fn {
        inline void operator()(FILE* f) const {
            if (f) fclose(f);
        }
    };

    /** \brief Smart pointer to FILE that automatically calls fclose().
     */
    using file_ptr = std::unique_ptr<FILE,fclose_fn>;


    /** \brief Open file for reading binary and return auto-fclose pointer.
     *
     * The returned FILE* is wrapped in a unique_ptr that will automatically
     * call fclose() when destructed.
     *
     * On Windows, this method uses _wfopen() so it should properly handle
     * unicode path names.
     * On all other platforms, fopen() is used.
     */
    template <typename PATH>
    inline auto fopen_rb(const PATH& path) {
        using namespace wfopen_rb;
        return file_ptr(_wfopen(c_str(path), L"rb"));
    }


    /** \brief Open file for writing binary data.
     */
    template <typename PATH>
    inline auto fopen_wb(const PATH& path) {
        using namespace wfopen_wb;
        return file_ptr(_wfopen(c_str(path), L"wb"));
    }


    /** \brief Type trait to identify path objects that are acceptable to
     * the above fopen methods.
     *
     * Generally, any path object for which c_str(path) returns const char*.
     * Also, on Windows only, path objects for which c_str(path)
     * returns const wchar_t*.
     */
    template <typename PATH, typename = void>
    struct is_fopen_path : std::false_type {};

    template <typename PATH>
    constexpr auto wfopen_tester() {
        using namespace wfopen_rb;
        return _wfopen(c_str(PATH{}), L"rb");
    }
    template <typename PATH>
    struct is_fopen_path<PATH,std::enable_if_t<std::is_same_v<decltype(wfopen_tester<std::decay_t<PATH> >()), FILE*> > >
        : std::true_type {};

    template <typename PATH>
    constexpr auto is_fopen_path_v = is_fopen_path<PATH>::value;
}
