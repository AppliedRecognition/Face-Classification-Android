
#include <iostream>

#include "ostream_sink.hpp"

using namespace applog;

ostream_sink::shared_ptr ostream_sink::add_sink(std::ostream& out,
                                                log_level base_level) {
    shared_ptr sink =
        std::make_shared<ostream_sink>(out,base_level);
    add_sink(sink);
    return sink;
}

void ostream_sink::write_log(const std::string& log_line, 
                             bool day_msg, bool) {
    if (day_msg)
        pending_day_msg = log_line;
    else {
        if (!pending_day_msg.empty()) {
            out << pending_day_msg << std::flush;
            pending_day_msg.clear();
        }
        out << log_line << std::flush;
    }
}
