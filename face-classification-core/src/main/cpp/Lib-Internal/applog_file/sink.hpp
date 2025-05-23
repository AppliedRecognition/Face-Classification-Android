#pragma once

#include <applog/filter_sink.hpp>

#include <stdext/options_tuple.hpp>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>


namespace applog {

    /** \brief Max files option.
     *
     * Default value if not specified is 90.
     * If zero, then all log files except the one currently being written to
     * will be deleted.
     */
    enum class max_files : unsigned {};

    /** \brief Continuous file option.
     *
     * Without this option, the default behaviour is to start a new log file
     * at midnight each day.
     */
    struct continuous_tag;
    using continuous_option = stdx::option_bool<continuous_tag>;
    const continuous_option continuous{true};


    class file_sink final : public filter_sink {
    public:
        using shared_ptr = std::shared_ptr<file_sink>;
        using options_type =
            stdx::options_tuple<log_level,applog::max_files,continuous_option>;
        static shared_ptr add_sink_with_opts(std::filesystem::path directory,
                                             std::string prefix,
                                             const options_type& opts);

        /** \brief Construct and activate new sink.
         *
         * Default options are: <ul>
         *   <li>minimum level is logTRACE (log everything)</li>
         *   <li>maximum files is 90</li>
         *   <li>start new log file each day (not continuous)</li>
         * </ul>
         *
         * Special case: if prefix is a complete filename with extension ".log"
         * and continuous is specified as an option, then this single file
         * is created as the log file.
         * In this case, if the file already exists an exception is thrown.
         * Also in this case, no files are ever deleted regardless of the
         * maximum files setting.
         */
        template <typename... Opts>
        static inline shared_ptr add_sink(std::filesystem::path directory,
                                          std::string prefix, Opts&&... opts) {
            return add_sink_with_opts(
                directory, prefix,
                { logTRACE, applog::max_files(90),
                        std::forward<Opts>(opts)... });
        }
        using sink::add_sink;
        

        /** \brief Constructor.
         *
         * Use add_sink() instead.
         */
        file_sink(std::filesystem::path directory,
                  std::string prefix, const options_type& opts);
        ~file_sink();

        inline const std::string& current_file() const {
            return m_current_file;
        }
        inline const std::set<std::string>& session_files() const {
            return m_session_files;
        }
        std::set<std::string> all_files(const std::string& prefix) const;
        inline std::set<std::string> all_files() const {
            return all_files(prefix);
        }

        const std::filesystem::path directory;
        const std::string prefix;
        const unsigned max_files;

    private:
        static bool is_ours(const std::filesystem::path& filename,
                            const std::string& prefix);
        void open_new_file();
        bool prune_excess();
        void write_log(const std::string& log_line, 
                       bool day_msg, bool new_day) override;

        bool m_continuous;
        std::string m_current_file;
        std::set<std::string> m_session_files;

        std::unique_ptr<std::ofstream> out;

        using lock_type = std::lock_guard<std::mutex>;
        std::mutex m_file_mutex;
    };

}


