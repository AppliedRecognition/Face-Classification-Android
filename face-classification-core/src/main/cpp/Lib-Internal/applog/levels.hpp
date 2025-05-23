#pragma once

namespace applog {

    /** \brief Log levels.
     *
     * These levels are taken from 
     * <a href="http://en.wikipedia.org/wiki/Log4j">Log4j</a>.
     * <dl>
     *   <dt>FATAL</dt>
     *     <dd>Severe errors that cause premature termination.</dd>
     *   <dt>ERROR</dt>
     *     <dd>Runtime errors or unexpected conditions that do not cause
     *         immediate termination of the application.</dd>
     *   <dt>WARNING</dt>
     *     <dd>Use of deprecated APIs, poor use of API, 'almost' errors, other
     *         runtime situations that are undesirable or unexpected, but not
     *         necessarily "wrong".  High volume server applications may
     *         use this level as the default for logging.</dd>
     *   <dt>INFO</dt>
     *     <dd>The first of the "for debugging" information levels.
     *         Applications as run by users (including alpha and beta releases)
     *         or any other production environment should use this level as the 
     *         default for logging.  
     *         Events that only occur once (startup, shutdown, etc.) should
     *         be logged at this level.
     *         Events that are "per item" may be included at this level or
     *         at a lower level.</dd>
     *   <dt>DEBUG</dt>
     *     <dd>Deprecated due to confusion regarding its purpose.
     *         Do not use.</dd>
     *   <dt>DETAIL</dt>
     *     <dd>Detailed information on the flow through the system. 
     *         This level is the default for logging for the developers
     *         (ie. when they run the application).
     *         It is expected to be very detailed so as to generally provide
     *         all of the information necessary for development.</dd>
     *   <dt>TRACE</dt>
     *     <dd>Finest level of detail.  Only used when debugging a specific
     *         problem that cannot be resolved with the higher levels.
     *         Messages that are most likely not needed but should be
     *         retained nonetheless are logged at this level.</dd>
     * </dl>
     */
    enum log_level {
        logNONE    = -1,
        logFATAL   = 0,
        logERROR   = 1,
        logWARNING = 2,
        logINFO    = 3,
        logDEBUG   = 4,
        logDETAIL  = 5,
        logTRACE   = 6
    };

}

using applog::logNONE;
using applog::logFATAL;
using applog::logERROR;
using applog::logWARNING;
using applog::logINFO;
using applog::logDETAIL;
using applog::logTRACE;

#ifndef __FBLIB_APPLOG_REMOVE_DEPRECATED__
using applog::logDEBUG;
#endif
