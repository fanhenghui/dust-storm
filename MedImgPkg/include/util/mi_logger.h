#ifndef MEDIMGUTIL_MI_FILE_UTIL_H
#define MEDIMGUTIL_MI_FILE_UTIL_H

#include "util/mi_util_export.h"

#include "boost/shared_ptr.hpp"
#include "boost/thread/mutex.hpp"

#include "boost/log/sources/logger.hpp"
#include "boost/log/sources/global_logger_storage.hpp"
#include "boost/log/sources/severity_logger.hpp"
//#include "boost/log/utility/setup/console.hpp"
//#include "boost/log/utility/setup/file.hpp"
#include "boost/log/sinks/text_ostream_backend.hpp"
#include "boost/log/sinks/text_file_backend.hpp"
//#include "boost/log/sinks/basic_sink_frontend.hpp"
//#include "boost/log/sinks/basic_sink_backend.hpp"
#include "boost/log/sinks/sync_frontend.hpp"

namespace logging = boost::log;
namespace src = boost::log::sources;
namespace sinks = boost::log::sinks;
namespace expr = boost::log::expressions;  
namespace keywords = boost::log::keywords;
namespace attrs = boost::log::attributes;

namespace medical_imaging {
enum SeverityLevel {
    TRACE = 0,
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    FATAL,
};
}

template< typename CharT, typename TraitsT >
inline std::basic_ostream< CharT, TraitsT >& operator<< (
    std::basic_ostream< CharT, TraitsT >& strm, medical_imaging::SeverityLevel lvl)
{
    static const int lvlnum = 6; 
    static const char* const lvlstr[lvlnum] = {
        "Trace",
        "Debug",
        "Info",
        "Warning",
        "Error",
        "Fatal",
    };

    if (static_cast<int>(lvl) < lvlnum ) {
        strm << lvlstr[lvl];
    } else {
        strm << "Invalid severity";
    }
    return strm;
}

BOOST_LOG_INLINE_GLOBAL_LOGGER_DEFAULT(mi_logger,src::severity_logger_mt<medical_imaging::SeverityLevel>)

#ifndef BOOST_LOG_NO_THREADS
    typedef sinks::synchronous_sink<sinks::text_file_backend> FileSinkType;
    typedef sinks::synchronous_sink<sinks::basic_text_ostream_backend<char>> ConsoleSinkType;
#else
    typedef sinks::unlocked_sink<sinks::text_file_backend> FileSinkType;
    typedef sinks::unlocked_sink<sinks::basic_text_ostream_backend<char>> ConsoleSinkType;
#endif

#define MI_LOG(sev) BOOST_LOG_SEV(mi_logger , sev)

MED_IMG_BEGIN_NAMESPACE

class Util_Export Logger {
public:
    static Logger* instance();
    ~Logger();

    void filter_level_console(SeverityLevel level);
    void filter_level_file(SeverityLevel level);

private:
    Logger();
    void init_i();

private:
    static Logger* _s_instance;
    static boost::mutex _s_mutex;

    SeverityLevel _console_filer_level;
    SeverityLevel _file_filer_level;
    boost::shared_ptr<FileSinkType> _file_sink;
    boost::shared_ptr<ConsoleSinkType> _console_sink;
    
};

MED_IMG_END_NAMESPACE

#endif