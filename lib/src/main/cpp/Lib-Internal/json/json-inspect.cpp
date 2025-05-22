
#ifndef __LIB_INTERNAL_BUILD_APPLICATIONS__
#error json-inspect.cpp is an application -- not part of the json library
#endif

#include "io.hpp"
#include "io_manip.hpp"

#include <applog/applog.hpp>

#include <filesystem>
#include <iostream>


int main(int argc, char*argv[]) {
    const char* const prog = [&](){
        assert(argc > 0);
        if (auto p = strrchr(argv[0], '/')) return p + 1;
        return argv[0];
    }();
    --argc, ++argv;

    if (argc <= 0) {
        std::cout << "Usage:" << std::endl
                  << '\t' << prog << " files..." << std::endl;
        return 1;
    }

    for ( ; argc > 0; --argc, ++argv) {
        const auto path = std::filesystem::path(*argv);
        if (!is_regular_file(path)) {
            FILE_LOG(logERROR) << "file not found: " << path;
            continue;
        }
        const auto top = json::load(path);
        std::cout << path.stem() << ": "
                  << json::indent("    ")
                  << json::binary_subst("<BYTES:###>")
                  << top << std::endl;
    }


    return 0;
}
