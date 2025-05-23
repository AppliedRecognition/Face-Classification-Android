
#include "loader_impl.hpp"

#include <filesystem>
#include <fstream>

using namespace models;

using std::filesystem::path;
using std::filesystem::directory_iterator;
using std::ifstream;

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
