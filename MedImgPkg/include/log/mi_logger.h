#ifndef MEDIMGLOG_MI_LOGGER_H
#define MEDIMGLOG_MI_LOGGER_H

#include <string>
#include <ostream>
#include <iomanip>

#include "log/mi_logger_export.h"
#include "log/mi_logger_define.h"

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
    Log_Export void set_file_direction(const std::string& tag_dir = std::string("logs"));
    Log_Export void set_file_name_format(const std::string& name_format = std::string("mi-%Y-%m-%d_%H-%M-%S.%N.log"));
    Log_Export void set_file_max_size(unsigned int size = 32*1024*1024);
    Log_Export void set_file_rotation_size(unsigned int size);
    Log_Export void set_file_rotation_time(unsigned char hour = 0, unsigned char min = 0, unsigned char sec = 0);
    Log_Export void set_min_free_space(unsigned int free_space = 100*1024*1024);
    //sink filter
    Log_Export void filter_level_stream(SeverityLevel level = MI_TRACE);
    Log_Export void filter_level_file(SeverityLevel level = MI_TRACE);

private:
    Logger();
    void read_config_file_i();

private:
    static Logger* _s_instance;
    static boost::mutex _s_mutex;
    bool _init;

    //struct InnerSink;
    //std::unique_ptr<InnerSink> _inner_sink;

    SeverityLevel _stream_filer_level;
    SeverityLevel _file_filer_level;

    std::string _file_target_dir;
    std::string _file_name_format;
    unsigned int _min_free_space;
    unsigned int _max_file_size;

    unsigned int _rotation_size;
    struct TimeHMS {
        unsigned char hour;
        unsigned char min;
        unsigned char sec;
        TimeHMS(unsigned char h , unsigned char m , unsigned char s ) : hour(h), min(m), sec(s) {}
        TimeHMS() : hour(0), min(0), sec(0) {}
    };
    TimeHMS _rotation_time;

    std::string _config_file;
};

MED_IMG_END_NAMESPACE

#endif