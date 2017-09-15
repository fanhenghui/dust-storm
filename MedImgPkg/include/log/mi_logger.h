#ifndef MEDIMGLOG_MI_LOGGER_H
#define MEDIMGLOG_MI_LOGGER_H

#include <string>
#include <ostream>
#include <iomanip>

#include "log/mi_logger_export.h"

#include "boost/thread/mutex.hpp"

#include "boost/date_time/posix_time/posix_time.hpp"
#include "boost/log/core.hpp"
#include "boost/log/expressions.hpp"
#include "boost/log/attributes.hpp"
#include "boost/log/sources/basic_logger.hpp"
#include "boost/log/sources/severity_logger.hpp"
#include "boost/log/sources/record_ostream.hpp"
#include "boost/log/sources/global_logger_storage.hpp"

namespace logging = boost::log;
namespace src = boost::log::sources;
namespace expr = boost::log::expressions;
namespace sinks = boost::log::sinks;
namespace attrs = boost::log::attributes;
namespace keywords = boost::log::keywords;

namespace medical_imaging{
enum SeverityLevel {
    MI_TRACE,
    MI_DEBUG,
    MI_INFO,
    MI_WARNING,
    MI_ERROR,
    MI_FATAL,
};

template< typename CharT, typename TraitsT >
inline std::basic_ostream< CharT, TraitsT >& operator<< ( std::basic_ostream< CharT, TraitsT >& strm, SeverityLevel lvl) {
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
}


BOOST_LOG_INLINE_GLOBAL_LOGGER_DEFAULT(mi_logger,src::severity_logger< medical_imaging::SeverityLevel >)

BOOST_LOG_ATTRIBUTE_KEYWORD(line_id, "LineID", unsigned int)
BOOST_LOG_ATTRIBUTE_KEYWORD(severity, "Severity", medical_imaging::SeverityLevel)
BOOST_LOG_ATTRIBUTE_KEYWORD(scope, "Scope", attrs::named_scope::value_type)
BOOST_LOG_ATTRIBUTE_KEYWORD(timeline, "Timeline", attrs::timer::value_type)
BOOST_LOG_ATTRIBUTE_KEYWORD(thread_id, "ThreadID", boost::log::aux::thread::id)

#define MI_LOG(sev) BOOST_LOG_SEV(mi_logger::get() , sev)
#define MI_DETAIL_FORMAT "[File: " << __FILE__ << "; Func: " << __FUNCTION__ << "; Line: " __LINE__ << "] "

MED_IMG_BEGIN_NAMESPACE

class Logger {
public:
    Log_Export static Logger* instance();
    Log_Export ~Logger();

    Log_Export void initialize();

    Log_Export void bind_config_file(const std::string& file_name);
    //text file log sink
    Log_Export void set_target(const std::string& tag_dir);
    Log_Export void set_file_name_format(const std::string& name_format = std::string("sign-%Y-%m-%d_%H-%M-%S.%N.log"));
    Log_Export void set_rotation_size(unsigned int size);
    Log_Export void set_time_based_rotation(unsigned char hour , unsigned char min , unsigned char sec);
    Log_Export void set_free_space(unsigned int free_space);
    //sink filter
    Log_Export void filter_level_stream(SeverityLevel level);
    Log_Export void filter_level_file(SeverityLevel level);

private:
    Logger();

private:
    static Logger* _s_instance;
    static boost::mutex _s_mutex;

    //struct InnerSink;
    //std::unique_ptr<InnerSink> _inner_sink;

    SeverityLevel _stream_filer_level;
    SeverityLevel _file_filer_level;
    std::string _target_dir;
    std::string _file_name_format;
    unsigned int _rotation_size;
    unsigned int _min_free_space;

    struct TimeHMS {
        unsigned char _hour;
        unsigned char _min;
        unsigned char _sec;
        TimeHMS(unsigned char h , unsigned char m , unsigned char s ) : _hour(h), _min(m), _sec(s) {}
        TimeHMS() : _hour(0), _min(0), _sec(0) {}
    };
    TimeHMS _time_based_rotation;
};


MED_IMG_END_NAMESPACE

#endif