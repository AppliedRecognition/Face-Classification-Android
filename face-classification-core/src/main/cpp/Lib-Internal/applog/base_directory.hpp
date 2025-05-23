#pragma once

#include <string>
#include <filesystem>


/** \brief Find the specified directory in the current directory or any
 * parent directory.
 *
 * Walk up the directory tree, starting at the current working directory, 
 * looking for a directory with the specified name.  
 * If such a directory is not found, simply return the directory as if it
 * were found in the current working directory.
 *
 * \param[in] dir name of directory to look for
 * \return relative path from current working directory to dir
 */
inline auto base_directory(std::filesystem::path dir) {
    auto result = dir;
    for (unsigned depth = 0; depth < 10; ++depth) {
        if (is_directory(result))
            return result;
        result = ".." / result;
    }
    return dir;  // failed: just return dir name
}
