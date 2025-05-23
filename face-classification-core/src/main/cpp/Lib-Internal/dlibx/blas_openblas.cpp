
/* Note:
 * This file is mutually exclusive with blas_other.cpp.
 * Only compile and link one of them.
 * If not using openblas, then link to blas_other.cpp (not this file).
 */

#include "library_init.hpp"
#include <applog/core.hpp>
#include <sstream>

#if __has_include(<openblas/cblas.h>)
#include <openblas/cblas.h>
#else
#include <cblas.h>
#endif

void dlibx::library_init_rec::openblas_init() {
    std::stringstream ss;
    ss << "openblas: " << openblas_get_config();

    // openblas_get_corename() is part of config
        
    switch (const auto par = openblas_get_parallel()) {
    case OPENBLAS_SEQUENTIAL: ss << " sequential"; break;
    case OPENBLAS_THREAD:     ss << " multi-threaded"; break;
    case OPENBLAS_OPENMP:     ss << " OpenMP"; break;
    default: ss << " unknown_threading(" << par << ")";
    }
    ss << ' ' << openblas_get_num_procs() << " cores";

    openblas_set_num_threads(1);
    const auto t = openblas_get_num_threads();
    if (t != 1)
        FILE_LOG(logWARNING) << "openblas: failed to set number of threads";
    
    ss << ' ' << t << " thread";
    FILE_LOG(logINFO) << ss.str();
}
