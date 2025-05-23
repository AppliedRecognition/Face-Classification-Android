
#include "sink.hpp"
#include <applog/time_point.hpp>

#include <cassert>
#include <set>
#include <vector>


using namespace applog;


file_sink::shared_ptr
file_sink::add_sink_with_opts(std::filesystem::path directory,
                              std::string prefix,
                              const options_type& opts) {
    auto sink =
        std::make_shared<file_sink>(std::move(directory), move(prefix), opts);
    add_sink(sink);
    return sink;
}

file_sink::file_sink(std::filesystem::path directory, std::string prefix,
                     const options_type& opts)
    : filter_sink(std::get<log_level>(opts)),
      directory(std::move(directory)),
      prefix(move(prefix)),
      max_files(unsigned(std::get<applog::max_files>(opts))),
      m_continuous(std::get<continuous_option>(opts)) {
    if (m_continuous) {
        const auto n = this->prefix.size();
        if (n > 4 && this->prefix.compare(n-4,4,".log") == 0) {
            // special case
            auto fn = this->directory / this->prefix;
            if (exists(fn))
                throw std::runtime_error("log file exists -- not overwriting");
            out = std::make_unique<std::ofstream>(fn);
            return;
        }
    }
    //general case
    open_new_file();
    prune_excess();
}

file_sink::~file_sink() {
    if (out) {
        (*out) << "--" << std::endl;
        out->close();
        out.reset();
    }
}

void file_sink::open_new_file() {
    const auto date_str = now().utc_iso_string();
    if (!m_current_file.empty())
        m_session_files.insert(m_current_file);
    m_current_file = prefix + date_str + ".log";
    out = std::make_unique<std::ofstream>(directory/m_current_file);
}

bool file_sink::is_ours(const std::filesystem::path& filename,
                        const std::string& prefix) {
    const auto s = filename.generic_string();
    return s.size() >= prefix.size() + 4 && 
        s.compare(0,prefix.size(),prefix) == 0 &&
        s.compare(s.size()-4,4,".log") == 0;
}

bool file_sink::prune_excess() {
    std::vector<std::filesystem::path> to_remove;
    try {
        std::set<std::filesystem::path> files;
        using std::filesystem::directory_iterator;
        for (directory_iterator it(directory), end; it != end; ++it) {
            if (!is_regular_file(it->status()))
                continue;
            const auto& filename = it->path().filename();
            if (!is_ours(filename,prefix)) continue;
            if (files.size() < max_files)
                files.insert(filename);
            else if (filename > *files.begin()) {
                to_remove.push_back(directory / *files.begin());
                files.erase(files.begin());
                files.insert(filename);
                assert(files.size() == max_files);
            }
            else
                to_remove.push_back(directory / filename);
        }
    }
    catch (const std::exception&) {
        return false;
    }
    bool good = true;
    for (const auto& x : to_remove) {
        try {
            remove(x);
        }
        catch (const std::exception&) {
            good = false;
        }
    }
    return good;
}

std::set<std::string> file_sink::all_files(const std::string& prefix) const {
    std::set<std::string> files;
    using std::filesystem::directory_iterator;
    for (directory_iterator it(directory), end; it != end; ++it) {
        if (!is_regular_file(it->status()))
            continue;
        const auto& fn = it->path().filename();
        if (is_ours(fn,prefix))
            files.insert(fn.generic_string());
    }
    return files;
}

void file_sink::write_log(const std::string& log_line, bool, bool new_day) {
    lock_type lock(m_file_mutex);
    if (new_day && !m_continuous) {
        if (out) {
            (*out) << "--" << std::endl;
            out->close();
            out.reset();
        }
        open_new_file();
        if (!prune_excess()) {
            // todo: log error or something
        }
    }
    if (out) (*out) << log_line << std::flush;
}

