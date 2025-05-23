#pragma once

#include <string>
#include <exception>
#include <boost/current_function.hpp>

namespace applog {

    /** \brief Exception thrown by APPLOG_CHECK.
     */
    class check_failure : public std::exception {
    public:
        [[noreturn]] static void handle_assert(
            const char* expr, const char* function, 
            const char* file, long line);
        [[noreturn]] static void handle_check(
            const char* expr, const char* function, 
            const char* file, long line);
        check_failure(const std::string& msg) : msg(msg) {}
        ~check_failure() throw() {}
        const char* what() const throw() override {
            return msg.c_str();
        }

    private:
        std::string msg;
    };

}


/** \brief Check expression and throw exception if false.
 */
#define APPLOG_CHECK(expr) ((expr) ? ((void)0) :                      \
    applog::check_failure::handle_check(#expr,BOOST_CURRENT_FUNCTION, \
                                        __FILE__,__LINE__))

/** \brief Check expression and throw exception if false.
 */
#define AR_CHECK(expr) ((expr) ? ((void)0) :                          \
    applog::check_failure::handle_check(#expr,BOOST_CURRENT_FUNCTION, \
                                        __FILE__,__LINE__))


/** \brief Check expression and assert if false.
 */
#define APPLOG_ASSERT(expr) ((expr) ? ((void)0) :                      \
    applog::check_failure::handle_assert(#expr,BOOST_CURRENT_FUNCTION, \
                                         __FILE__,__LINE__))

/** \brief Check expression and assert if false.
 */
#define AR_ASSERT(expr) ((expr) ? ((void)0) :                          \
    applog::check_failure::handle_assert(#expr,BOOST_CURRENT_FUNCTION, \
                                         __FILE__,__LINE__))


