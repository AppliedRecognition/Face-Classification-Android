
#include "assert.hpp"
#include "logger.hpp"

#include <cassert>
#include <cstdlib>
#include <sstream>


using namespace applog;

    
void check_failure::handle_assert(const char* expr, const char* function,
                                  const char* file, long line) {
    APPLOG(logFATAL) << "ASSERT FAILED [" << file << ":" << line << " "
                     << function << "] " << expr;
    assert(!"ASSERT FAILED (see log file for details)");
    std::abort(); // in case assert is compiled out
}

void check_failure::handle_check(const char* expr, const char* function,
                                 const char* file, long line) {
    std::stringstream s;
    s << "CHECK FAILED [" << file << ":" << line << " " << function << "] "
      << expr;
    APPLOG(logERROR) << s.str();
    throw check_failure(s.str());
}
