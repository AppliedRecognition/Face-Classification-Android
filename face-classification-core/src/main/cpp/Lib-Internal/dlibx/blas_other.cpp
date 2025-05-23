
/* Note:
 * This file is mutually exclusive with blas_openblas.cpp.
 * Only compile and link one of them.
 * If using openblas, then link to blas_openblas.cpp (not this file).
 */

#include "library_init.hpp"
#include <applog/core.hpp>

void dlibx::library_init_rec::openblas_init() {
    FILE_LOG(logDETAIL) << "dlibx: openblas not initialized or not being used";
}
