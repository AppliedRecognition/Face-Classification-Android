
#include "loader_impl.hpp"

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

using namespace models;

using boost::filesystem::path;
using boost::filesystem::directory_iterator;
using boost::filesystem::ifstream;

static_assert(stdx::is_path_v<path>);

template <>
istream_ptr models::open_binary_file<path>(const path& fn) {
    return std::make_unique<ifstream>(fn, std::ios::binary);
}

template <>
loader_return_type
loader<path>::operator()(format f, type t, std::string_view name) const {
    return open_files(find_files<directory_iterator>(f, t, name));
}
