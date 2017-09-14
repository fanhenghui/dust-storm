#ifndef MEDIMGUTIL_MI_FILE_UTIL_H
#define MEDIMGUTIL_MI_FILE_UTIL_H

#include "util/mi_util_export.h"

#include "boost/thread/mutex.hpp"

#include <cstddef>
#include <string>
#include <ostream>
#include <fstream>
#include <iomanip>

#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/weak_ptr.hpp>
#include <boost/smart_ptr/make_shared_object.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/sources/basic_logger.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <boost/log/attributes/scoped_attribute.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>

#include <boost/log/sources/global_logger_storage.hpp>
#include <boost/log/utility/setup/file.hpp>

#include <boost/core/null_deleter.hpp>

namespace logging = boost::log;
namespace src = boost::log::sources;
namespace expr = boost::log::expressions;
namespace sinks = boost::log::sinks;
namespace attrs = boost::log::attributes;
namespace keywords = boost::log::keywords;

namespace medical_imaging{
enum SeverityLevel {
    TRACE = 0,
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    FATAL
};
}

BOOST_LOG_INLINE_GLOBAL_LOGGER_DEFAULT(mi_logger,src::severity_logger< medical_imaging::SeverityLevel >)

BOOST_LOG_ATTRIBUTE_KEYWORD(line_id, "LineID", unsigned int)
BOOST_LOG_ATTRIBUTE_KEYWORD(severity, "Severity", medical_imaging::SeverityLevel)
BOOST_LOG_ATTRIBUTE_KEYWORD(tag_attr, "Module", std::string)
BOOST_LOG_ATTRIBUTE_KEYWORD(scope, "Scope", attrs::named_scope::value_type)
BOOST_LOG_ATTRIBUTE_KEYWORD(timeline, "Timeline", attrs::timer::value_type)

#define MI_LOG(sev) BOOST_LOG_SEV(mi_logger::get() , sev)

MED_IMG_BEGIN_NAMESPACE

class Util_Export Logger {
public:
    static Logger* instance();
    ~Logger();

    void bind_config_file(const std::string& file_name);

    void init();

    //text file log sink
    void set_target(const std::string& tag_dir);
    void set_file_name_format(const std::string& name_format = std::string("sign-%Y-%m-%d_%H-%M-%S.%N.log"));
    void set_rotation_size(unsigned int size);
    void set_time_based_rotation(unsigned char hour , unsigned char min , unsigned char sec);
    void set_free_space(unsigned int free_space);
    //sink filter
    void filter_level_stream(SeverityLevel level);
    void filter_level_file(SeverityLevel level);
    //custom module attribute register
    int register_logger_module(src::severity_logger<SeverityLevel>& lg, const std::string& module_name);

private:
    Logger();

private:
    static Logger* _s_instance;
    static boost::mutex _s_mutex;

    struct InnerSink;
    std::unique_ptr<InnerSink> _inner_sink;

    SeverityLevel _stream_filer_level;
    SeverityLevel _file_filer_level;
    std::string _target_dir;
    std::string _file_name_format;
    unsigned int _rotation_size;
    unsigned int _min_free_space;

    struct TimeHMS {
        unsigned char hour;
        unsigned char min;
        unsigned char sec;
        TimeHMS(unsigned char h , unsigned char m , unsigned char s ) : hour(h), min(m), sec(s) {}
        TimeHMS() : hour(0), min(0), sec(0) {}
    };
    TimeHMS _time_based_rotation;
};

#include "util/mi_logger.inl"

MED_IMG_END_NAMESPACE

#endif