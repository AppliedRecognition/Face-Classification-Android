#pragma once

#include <ostream>
#include "filter_sink.hpp"


namespace applog {

    class ostream_sink final : public filter_sink {
    public:
        using shared_ptr = std::shared_ptr<ostream_sink>;

        static shared_ptr add_sink(std::ostream& out,
                                   log_level base_level = logTRACE);
        using sink::add_sink;
    
        ostream_sink(std::ostream& out,
                     log_level base_level = logTRACE) noexcept
            : filter_sink(base_level), out(out) {
        }

    private:
        void write_log(const std::string& log_line,
                       bool day_msg, bool) override;
        std::ostream& out;
        std::string pending_day_msg;
    };


    /** \brief Sink to stderr (default log sink).
     *
     * If the sink to stderr does not already exist, then it will be
     * added provided that level is not logNONE.
     * If level is logNONE in this case, then nullptr is returned.
     *
     * If the sink was previously added, then the pointer to the existing
     * sink is returned.  In this case, the sink's log level is not changed.
     *
     * \param level only used when sink is created and added
     * \return nullptr or pointer to existing cerr sink
     */
    const ostream_sink::shared_ptr& cerr_sink(log_level level = logNONE);


    /** \brief Remove stderr sink (if it exists).
     */
    void remove_cerr_sink();
}

