#pragma once

#include "types.hpp"
#include <stdext/path_traits.hpp>

namespace models {

    /** \brief Open read-only binary file from path.
     *
     * Support for std::filesystem::path and boost::filesystem::path
     * are provided if the corresponding .cpp files are built.
     */
    template <typename PATH>
    istream_ptr open_binary_file(const PATH& path);


    /** \brief Model loader for files on disk in specified directory.
     *
     * Support for std::filesystem::path and boost::filesystem::path
     * are provided if the corresponding .cpp files are built.
     */
    template <typename PATH>
    class loader final {
        static_assert(stdx::is_path_v<PATH>,
                      "models::loader requires a filesystem path type class");

        template <typename DIRECTORY_ITERATOR>
        auto find_files(format f, type t, std::string_view name) const;

        static auto open_files(const std::vector<PATH>& fns);

    public:
        const PATH models_directory;

        /** \brief Constructor.
         */
        loader(PATH models_directory)
            : models_directory(std::move(models_directory)) {}

        /** \brief Loader function.
         */
        loader_return_type operator()(format, type, std::string_view) const;
    };
}
