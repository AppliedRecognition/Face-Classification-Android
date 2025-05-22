
#include "filter_sink.hpp"

using namespace applog;


log_level filter_sink::module_entered(const module& m, log_level level) const {
    if (level < base_level)
        level = base_level;
    module_levels_type::const_iterator it = module_levels.find(m);
    if (it != module_levels.end() && 
        level < it->second)
        level = it->second;
    return level;
}

