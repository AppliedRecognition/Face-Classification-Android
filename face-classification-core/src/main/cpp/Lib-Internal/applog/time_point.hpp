#pragma once

#include <array>
#include <chrono>
#include <string>

namespace applog {

    enum class day_number_type : int {};

    /** \brief Current UTC time and timezone offset in minutes.
     */
    struct time_point {
        const std::chrono::system_clock::time_point now;  // utc
        const int tzofs_minutes;

        /// "yyyymmddThhmmss"
        std::string utc_iso_string() const;

        /// e.g. "Fri, 22 Nov 2019"
        std::string local_day_string() const;

        /// local day of year
        day_number_type local_day_number() const;

        /// local time of day e.g. "14:24:38.662"
        std::array<char,13> local_time_of_day() const;
    };

    time_point now();
}
