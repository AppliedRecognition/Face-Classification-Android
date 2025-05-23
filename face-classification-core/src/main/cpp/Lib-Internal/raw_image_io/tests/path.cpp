#include <boost/test/unit_test.hpp>

#include <boost/filesystem.hpp>
#include <filesystem>

#include <stdext/stdio.hpp>


BOOST_AUTO_TEST_SUITE(raw_image)

BOOST_AUTO_TEST_CASE(generic_fopen_char) {
    const char* path0 = "asdf";
    const auto path1 = std::string(path0);
    const auto path2 = std::filesystem::path(path0);
    const auto path3 = boost::filesystem::path(path0);

    const auto c0 = stdx::c_str(path0);
    const auto c1 = stdx::c_str(path1);
    const auto c2 = stdx::c_str(path2);
    const auto c3 = stdx::c_str(path3);
    BOOST_CHECK_EQUAL(strcmp(c0,c1),0);
    BOOST_CHECK_EQUAL(strcmp(c0,c2),0);
    BOOST_CHECK_EQUAL(strcmp(c0,c3),0);

    const auto s0 = stdx::generic_string(path0);
    const auto s1 = stdx::generic_string(path1);
    const auto s2 = stdx::generic_string(path2);
    const auto s3 = stdx::generic_string(path3);
    BOOST_CHECK_EQUAL(s0,s1);
    BOOST_CHECK_EQUAL(s0,s2);
    BOOST_CHECK_EQUAL(s0,s3);

    if (0) {
        char* cc = nullptr;
        stdx::c_str(cc);
        stdx::generic_string(cc);
    }

    static_assert(stdx::is_fopen_path_v<decltype(path0)>);
    static_assert(stdx::is_fopen_path_v<decltype(path1)>);
    static_assert(stdx::is_fopen_path_v<decltype(path2)>);
    static_assert(stdx::is_fopen_path_v<decltype(path3)>);

    if (0) {
        stdx::fopen_rb(path0);
        stdx::fopen_rb(path1);
        stdx::fopen_rb(path2);
        stdx::fopen_rb(path3);
    }
}

BOOST_AUTO_TEST_CASE(generic_fopen_wchar) {
    const wchar_t* path0 = L"wasdf";
    const auto path1 = std::wstring(path0);

    const auto c0 = stdx::c_str(path0);
    const auto c1 = stdx::c_str(path1);
    BOOST_CHECK(std::wstring(c0) == std::wstring(c1));

    const auto s0 = stdx::generic_string(path0);
    const auto s1 = stdx::generic_string(path1);
    BOOST_CHECK_EQUAL(s0,std::string("wasdf"));
    BOOST_CHECK_EQUAL(s1,std::string("wasdf"));

    if (0) {
        wchar_t* cc = nullptr;
        stdx::c_str(cc);
        stdx::generic_string(cc);
    }

    /* these will only work on windows:
    static_assert(stdx::is_fopen_path_v<decltype(path0)>);
    static_assert(stdx::is_fopen_path_v<decltype(path1)>);
    stdx::fopen_rb(path0);
    stdx::fopen_rb(path1);
    */
}

BOOST_AUTO_TEST_SUITE_END()
