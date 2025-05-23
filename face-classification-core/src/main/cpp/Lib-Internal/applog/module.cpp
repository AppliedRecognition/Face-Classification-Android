
#include "module.hpp"
#include "logger.hpp"

#define __FBLIB_APPLOG_PRIVATE_INTERNAL_USE_ONLY__
#include "internal.hpp"


using namespace applog;



/**************** struct module::detail ****************/

std::mutex module::detail::number_mutex;
std::map<std::string,unsigned long long> module::detail::number_map;

module::detail::detail(std::string description, int flags)
    : description(description), flags(flags) {
    if (flags & flagNUMBER) {
        lock_type lock(number_mutex);
        std::stringstream ss;
        ss << number_map[description]++;
        this->description += ss.str();
    }
}

bool module::detail::find_ancestor(const module_shared_ptr& ancestor) const {
    for (const auto& p : parents)
        if (p == ancestor || p->find_ancestor(ancestor))
            return true;
    return false;
}

log_level module::detail::enter_parents(sink::shared_ptr sink, 
                                        log_level level) const {
    for (const auto& p : parents) {
        level = p->enter_parents(sink,level);
        level = sink->module_entered(p,level);
    }
    return level;
}

log_level module::detail::enter(const module_shared_ptr& module,
                                sink::shared_ptr sink, 
                                log_level prev_level) {
    return sink->module_entered(
        module, module->enter_parents(sink,prev_level));
}
log_level module::detail::enter(const module& module,
                                sink::shared_ptr sink, 
                                log_level prev_level) {
    return enter(module.state,sink,prev_level);
}

void module::detail::insert_parents(std::set<module_shared_ptr>& dest,
                                    const module_shared_ptr& module) {
    dest.insert(module->parents.begin(),module->parents.end());
}


/**************** class module ****************/

module::module(const std::string& description, int flags)
    : state(std::make_shared<detail>(description,flags)) {
}

module::module() 
    : state(std::make_shared<detail>()) {
}

const std::string& module::get_description() {
    return state->get_description();
}

void module::set_description(const std::string& description) {
    if (const auto internal = applog::internal::global::get()) {
        const auto lock = internal->get_unique_lock();
        state->set_description(description);
    }
    else
        state->set_description(description);
}

void module::register_submodule(module submodule) {
    if (state == submodule.state || state->find_ancestor(submodule.state)) {
        FILE_LOG(logERROR) << "module: cycle in hierarchy";
        throw std::logic_error("module: cycle in hierarchy");
    }
    if (const auto internal = applog::internal::global::get()) {
        const auto lock = internal->get_unique_lock();
        submodule.state->insert_parent(state);
        if (auto rec = applog::internal::thread::get())
            rec->invalidate_levels();
    }
    else
        submodule.state->insert_parent(state);
}


/**************** class section ****************/

std::string section::thread_push_back(
    const std::shared_ptr<module::detail>& ptr, log_level level) {
    if (auto rec = internal::thread::get())
        return rec->push_back(ptr, level);
    if (logNONE < APPLOG_MINIMUM_LEVEL)
        throw std::runtime_error("logging not available for section entry");
    return {}; // logging disabled
}

section::section(module m, log_level level) 
    : module_ptr(m.state),
      module_type(thread_push_back(module_ptr, level)),
      level(level) {
}

section::~section() {
    if (!module_type.empty())
        FILE_LOG(level) << module_type << " leave";
}


