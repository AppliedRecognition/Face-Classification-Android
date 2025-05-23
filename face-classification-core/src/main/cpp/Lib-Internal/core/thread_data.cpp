
#include "thread_data.hpp"

#include <applog/core.hpp>

using namespace core;

thread_data::thread_data(object_store<true>& global,
                         object_store<true>& context,
                         bool register_thread)
    : context_data{global,context},
      section(register_thread ? 
              new applog::section(
                  applog::module(
                      "CORE", applog::flagTHREAD|applog::flagNUMBER),
                  logINFO) : nullptr) {
}

thread_data::~thread_data() = default;
