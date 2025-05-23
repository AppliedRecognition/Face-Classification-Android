#include <boost/test/unit_test.hpp>
#include "core.hpp"
#include "ostream_sink.hpp"

boost::unit_test::test_suite* init_unit_test_suite(int,char*[]);
boost::unit_test::test_suite* init_unit_test_suite(int ac, char *av[]) {
    boost::unit_test::framework::master_test_suite().p_name.value = 
        "Internal Libs Test Suite";
    if (ac > 0) --ac, ++av;
    for ( ; ac > 0; --ac, ++av) {
        if (strcmp(*av, "detail") == 0)
            applog::filter_sink::set_base_level(applog::cerr_sink(), logDETAIL);
        else if (strcmp(*av, "trace") == 0)
            applog::filter_sink::set_base_level(applog::cerr_sink(), logTRACE);
        else {
            FILE_LOG(logWARNING) << "unrecognized command line arg: '"
                                 << *av << "'";
            FILE_LOG(logINFO) << "use 'detail' or 'trace' to set log level";
        }
    }
    return 0;
}
