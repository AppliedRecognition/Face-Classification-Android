
#include "library_init.hpp"
#include "matrix_ops.hpp"
#include <applog/core.hpp>

namespace dlibx {
    library_init_rec library_init;
}

void dlibx::library_init_rec::init() {
    openblas_init();
    FILE_LOG(logINFO) << "dlibx: platform " << ops::machine.description;
}
