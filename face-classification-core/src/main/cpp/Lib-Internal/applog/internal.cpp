
#include "module.hpp"
#include "logger.hpp"
#include "ostream_sink.hpp"

#define __FBLIB_APPLOG_PRIVATE_INTERNAL_USE_ONLY__
#include "internal.hpp"

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <locale>
#include <string_view>


/// name of main thread
#ifndef APPLOG_MAIN_THREAD
#define APPLOG_MAIN_THREAD "MAIN"
#endif

/// log level for main thread enter and leave messages
#ifndef APPLOG_MAIN_LEVEL
#define APPLOG_MAIN_LEVEL logINFO
#endif

/// default log level for output to std::cerr
/// note: use logNONE to not add the sink
#ifndef APPLOG_CERR_LEVEL
#define APPLOG_CERR_LEVEL logINFO
#endif


namespace {
#if defined(_MSC_VER)
    struct gmtime_t {
        struct tm* ptr;
        gmtime_t(time_t t) : ptr(gmtime(&t)) {}
    };
    struct localtime_t {
        struct tm* ptr;
        localtime_t(time_t t) : ptr(localtime(&t)) {}
    };
#else
    struct gmtime_t {
        tm buf;
        tm* ptr;
        gmtime_t(time_t t) : ptr(gmtime_r(&t,&buf)) {}
    };
    struct localtime_t {
        tm buf;
        tm* ptr;
        localtime_t(time_t t) : ptr(localtime_r(&t,&buf)) {}
    };
#endif
}


using namespace applog;


/**************** struct internal::thread ****************/

internal::thread::thread(std::shared_ptr<internal::global> gp,
                         std::atomic<bool>* available_flag)
    : global(move(gp)),
      available_flag(available_flag),
      m_instance_iter(m_instance.end()) {
    assert(global);
    m_modules.push_back(global->base_module.state);
    std::stringstream ss;
    ss << this;
    m_thread_iterator = global->add_thread(ss.str());
}

internal::thread::~thread() {
    thread_section.reset();
    global->erase_thread(m_thread_iterator);
    if (available_flag)
        available_flag->store(false, std::memory_order_release);
}

logger::detail* internal::thread::enter() {
    if (m_instance_iter == m_instance.end()) {
        m_instance.emplace_back();
        return &m_instance.back();
    }
    else return &*m_instance_iter++;
}

void internal::thread::leave(logger::detail* ptr) {
    if (ptr) {
        assert(m_instance_iter != m_instance.begin());
        --m_instance_iter;
        assert(ptr == &*m_instance_iter);
        ptr->reset(global->base_stream);
    }
}

const std::string& internal::thread::thread_name() {
    if (m_thread_name.empty()) {
        std::stringstream ss;
        ss << std::hex << (std::size_t(this) / sizeof(thread));
        m_thread_name = ss.str();
        if (m_thread_name.size() >= 8)
            m_thread_name = m_thread_name.substr(m_thread_name.size()-7);
        m_thread_name.insert(m_thread_name.begin(), 'x');
        m_thread_name.resize(8,' ');
    }
    return m_thread_name;
}

const std::string& internal::thread::module_tags() {
    if (!m_module_tags) {
        m_module_tags.emplace();
        m_module_tag_set.clear();
        for (const auto& weak : m_modules) {
            auto m = weak.lock();
            if (m && !m->is_thread() &&
                m_module_tag_set.count(m) == 0) {
                std::string desc = m->get_description();
                if (!desc.empty()) {
                    desc += ' ';
                    *m_module_tags += desc;
                    module::detail::insert_parents(m_module_tag_set,m);
                    m_module_tag_set.insert(move(m));
                }
            }
        }
    }
    return *m_module_tags;
}

std::string internal::thread::extra_module_tag(const module_shared_ptr& m) {
    if (!m_module_tags)
        module_tags();
    std::string result;
    if (m_module_tag_set.count(m) == 0) {
        result = m->get_description();
        if (!result.empty())
            result += ' ';
    }
    return result;
}
std::string internal::thread::extra_module_tag(const module& m) {
    return extra_module_tag(m.state);
}

const char* internal::thread::push_back(const module_shared_ptr& m,
                                             log_level enter_level) {
    if (m->is_thread()) {
        if (!m_thread_module.expired()) {
            FILE_LOG(logERROR) << "additional thread module entered";
            throw std::logic_error("cannot enter two thread modules in single thread");
        }
        m_thread_module = m;
        m_modules.push_back(m);
        const std::string old_name = m_thread_name;
        m_thread_name = m->get_description();
        if (!m_thread_name.empty()) {
            if (m_thread_name.size() < 8)
                m_thread_name.resize(8,' ');
            if (!old_name.empty() && old_name != m_thread_name)
                FILE_LOG(enter_level) << "thread enter (rename from "
                                      << old_name << ")";
            else
                FILE_LOG(enter_level) << "thread enter";
            *m_thread_iterator = m_thread_name;
            return "thread";
        }
    }
    else {
        m_modules.push_back(m);
        std::string desc = m->get_description();
        if (!desc.empty()) {
            if (!m_module_tags)
                module_tags();
            if (m_module_tag_set.count(m) == 0) {
                desc += ' ';
                *m_module_tags += desc;
                module::detail::insert_parents(m_module_tag_set,m);
                m_module_tag_set.insert(m);
                FILE_LOG(enter_level) << "section enter";
                return "section";
            }
        }
    }
    return "";
}

void internal::thread::update_sink_levels(
    const sink::shared_ptr& sink, std::vector<log_level>& levels) const {
    if (levels.size() > m_valid_levels)
        levels.resize(m_valid_levels);
    levels.reserve(m_modules.size());
    while (levels.size() < m_modules.size()) {
        std::shared_ptr<module::detail> ptr = m_modules[levels.size()].lock();
        if (!ptr)
            continue;
        if (levels.empty())
            levels.push_back(module::detail::enter(ptr,sink,logNONE));
        else
            levels.push_back(module::detail::enter(ptr,sink,levels.back()));
    }
}

void internal::thread::sink_levels_updated() {
    m_valid_levels = m_modules.size();
}

void internal::thread::invalidate_levels() {
    m_valid_levels = 0;
}

void internal::thread::cleanup() {
    // remove any expired sections
    while (!m_modules.empty() && m_modules.back().expired()) {
        m_modules.pop_back();
        m_module_tags = std::nullopt;
    }
    assert(!m_modules.empty());
    if (m_valid_levels > m_modules.size())
        m_valid_levels = m_modules.size();
}


/**************** struct global::sink_record ****************/

internal::global::sink_record::sink_record()
    : thread_levels(std::make_shared<std::list<levels_type> >()) {
}

void internal::global::sink_record::reset() {
    std::lock_guard<std::mutex> lock(m_levels_mutex);
    thread_levels = std::make_shared<std::list<levels_type> >();
}

const internal::global::sink_record::levels_type&
internal::global::sink_record::get_levels(
    const sink::shared_ptr& sink, thread& thread_rec) {

    assert(sink);
    auto& w = thread_rec.m_sink_levels[sink.get()];
    if (auto p = w.lock()) {
        thread_rec.update_sink_levels(sink,*p);
        assert(!p->empty());
        return *p;
    }

    std::lock_guard<std::mutex> lock(m_levels_mutex);
    thread_levels->emplace_back();
    const auto p =
        std::shared_ptr<levels_type>(thread_levels, &thread_levels->back());
    thread_rec.update_sink_levels(sink,*p);
    assert(!p->empty());
    w = p;
    return *p;
}

void internal::global::sink_record::check_day(
    const sink::shared_ptr& sink, const time_point& now) {

    // pre-condition: shared_lock on sink_mutex is held
    // note: we hope access to prev_day is reasonably atomic
    const auto current_day = now.local_day_number();
    if (prev_day == current_day)
        return;

    bool new_day;
    {
        std::lock_guard<std::mutex> lock(m_day_mutex);
        if (prev_day == current_day)
            return;
        new_day = prev_day != day_number_type{0};
        prev_day = current_day;
    }

    std::stringstream ss;
    ss << "- " << now.local_day_string() << std::endl;
    sink->write_log(ss.str(),true,new_day);
}


/**************** struct time_point ****************/

using clock_type = std::chrono::system_clock;


std::string time_point::utc_iso_string() const {
    const auto t = clock_type::to_time_t(now);
    const auto utc = gmtime_t(t);
    std::stringstream ss;
    ss << std::put_time(utc.ptr, "%Y%m%dT%H%M%S");
    return ss.str();
}

std::string time_point::local_day_string() const {
    const auto t = clock_type::to_time_t(now) + tzofs_minutes*60;
    const auto local = gmtime_t(t);
    std::stringstream ss;
    ss << std::put_time(local.ptr, "%a, %d %b %Y");
    return ss.str();
}

day_number_type time_point::local_day_number() const {
    using namespace std::chrono;
    const auto t = duration_cast<seconds>(now.time_since_epoch()).count();
    return day_number_type{int((t + tzofs_minutes*60) / (24*3600))};
}

std::array<char,13> time_point::local_time_of_day() const {
    using namespace std::chrono;
    const auto t = duration_cast<milliseconds>(now.time_since_epoch()).count();
    const auto tod = int((t + tzofs_minutes*(60*1000)) % (24*3600*1000));
    const auto ms = tod % 1000;
    const auto s = (tod / 1000) % 60;
    const auto m = (tod / (60*1000)) % 60;
    const auto h = tod / (3600*1000);
    static const auto digit = [](int x) { return char('0'+x); };
    return std::array<char,13> {
        digit(h/10), digit(h%10), ':',
        digit(m/10), digit(m%10), ':',
        digit(s/10), digit(s%10), '.',
        digit(ms/100), digit((ms/10)%10), digit(ms%10), 0
    };
}

static auto calc_tzofs(time_t t) {
    const auto utc = gmtime_t(t);
    const auto local = localtime_t(t);
    const auto tod_ofs =
        (local.ptr->tm_hour - utc.ptr->tm_hour)*3600 +
        (local.ptr->tm_min - utc.ptr->tm_min)*60 +
        local.ptr->tm_sec - utc.ptr->tm_sec;
    auto d = local.ptr->tm_yday - utc.ptr->tm_yday;
    if (d < -1) d = 1;
    else if (d > 1) d = -1;
    return (24*3600+30 + d*24*3600 + tod_ofs) / 60 - 1440;
}

time_point applog::now() {
    return internal::global::now(nullptr);
}


/**************** struct internal::global ****************/

static auto trim_space(std::string_view sv) {
    const std::locale& loc = std::locale();
    while (!sv.empty() && std::isspace(sv.front(), loc))
        sv.remove_prefix(1);
    while (!sv.empty() && std::isspace(sv.back(), loc))
        sv.remove_suffix(1);
    return sv;
}

std::string internal::global::report_threads() {
    std::lock_guard<std::mutex> lock(m_thread_mutex);
    std::string result;
    auto it = m_thread_list.begin();
    if (it != m_thread_list.end()) {
        result.insert(result.size(), trim_space(*it));
        while (++it != m_thread_list.end()) {
            result += ' ';
            result.insert(result.size(), trim_space(*it));
        }
    }
    return result;
}

std::string applog::report_threads() {
    auto ptr = internal::global::get();
    return ptr ? ptr->report_threads() : std::string();
}


/* Static global and thread state.
 *
 * On startup:
 * Static POD (ie. pointers) are zero initialized before any code executes.
 * Static objects are constructed as they are encountered (in order).
 * Thread local objects (for main thread) are constructed all at once
 * (in order) the first time any of them are accessed.
 *
 * On shutdown:
 * Thread local objects (for main thread) are destructed in reverse order.
 * Then all static objects are destructed in reverse order.
 *
 * The thread object for the main thread (and only this one) will reset
 * logging_available to false when it is destructed.
 *
 * So logging is available after the main thread constructs the main_init
 * object (below), and logging is no longer available after the main
 * thread destroys the thread_local internal::thread object.
 *
 * Note that thread_ptr may not reset to nullptr after the object is
 * destructed so we cannot trust it's value when logging_available is false.
 *
 * The std::weak_ptr<internal::global> object is intentionally leaked
 * (never destructed).
 */

static std::weak_ptr<internal::global>* global_weak = nullptr;
static std::atomic<bool> logging_available;  // zero initialized
static thread_local std::unique_ptr<internal::thread> thread_ptr;

std::shared_ptr<internal::global> internal::global::get() {
    return global_weak ? global_weak->lock() : nullptr;
}

internal::thread* internal::thread::get() {
    if (global_weak && logging_available.load(std::memory_order_acquire)) {
        if (thread_ptr)
            thread_ptr->cleanup();
        else if (auto gp = global_weak->lock())
            thread_ptr = std::make_unique<internal::thread>(move(gp));
        else return nullptr;
        return thread_ptr.get();
    }
    return nullptr;
}

static auto& internal_cerr_sink() {
    static ostream_sink::shared_ptr p;
    return p;
}
void applog::remove_cerr_sink() {
    if (auto& p = internal_cerr_sink()) {
        sink::remove_sink(p);
        p.reset();
    }
}
const ostream_sink::shared_ptr& applog::cerr_sink(log_level level) {
    auto& p = internal_cerr_sink();
    if (logNONE < level && !p)
        p = ostream_sink::add_sink(std::cerr,level);
    return p;
}

static std::atomic<int> tzofs_epoch{0};
static std::atomic<int> tzofs_minutes{0};

time_point internal::global::now(global* ptr) {
    const auto now = clock_type::now();
    const auto t = clock_type::to_time_t(now);
    const auto epoch = int(t / 900);  // every 15 minutes
    if (epoch != tzofs_epoch.load(std::memory_order_acquire)) {
        std::shared_ptr<internal::global> shared;
        if (ptr ||
            (global_weak &&
             (ptr = (shared = global_weak->lock()).get()))) {
            std::lock_guard<std::mutex> lock(ptr->m_time_mutex);
            // have to check tzofs_epoch again
            if (epoch != tzofs_epoch.load(std::memory_order_acquire)) {
                tzofs_minutes.store(calc_tzofs(t), std::memory_order_release);
                tzofs_epoch.store(epoch, std::memory_order_release);
            }
        }
    }
    return time_point { now, tzofs_minutes.load(std::memory_order_acquire) };
}

namespace {
    struct initialize_global_and_main_thread {
        initialize_global_and_main_thread() {
            assert(!global_weak &&
                   !logging_available.load(std::memory_order_relaxed) &&
                   !thread_ptr);
            auto gp = std::make_shared<internal::global>();
            global_weak = new std::weak_ptr<internal::global>(gp);
            thread_ptr =
                std::make_unique<internal::thread>(gp, &logging_available);
            logging_available.store(true, std::memory_order_release);
            if (logNONE < APPLOG_MINIMUM_LEVEL)
                cerr_sink(APPLOG_CERR_LEVEL);
            thread_ptr->thread_section = std::make_unique<section>(
                module(APPLOG_MAIN_THREAD, flagTHREAD), APPLOG_MAIN_LEVEL);
        }
    };
}
static initialize_global_and_main_thread main_init;
