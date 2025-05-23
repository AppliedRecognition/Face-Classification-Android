#pragma once

#include <stdext/binary.hpp>
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>

namespace boostx {

    namespace internal {
        using namespace boost::interprocess;
        struct file_mapped_region {
            file_mapping fm;
            mapped_region r;
            template <typename U>
            explicit file_mapped_region(U&& fn)
                : fm(fn, read_only), r(fm, read_only) {
            }
        };
    }

    template <typename U>
    inline stdx::binary binary_file_mapping(U&& fn) {
        auto p = std::make_shared<internal::file_mapped_region>(fn);
        return stdx::binary({p,p->r.get_address()}, p->r.get_size());
    }
} 
