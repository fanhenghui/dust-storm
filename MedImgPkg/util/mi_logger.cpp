#include "mi_logger.h"

#include "boost/make_shared.hpp"
#include "boost/core/null_deleter.hpp"
#include "boost/thread.hpp"

#include "boost/log/core.hpp"

#include "boost/log/expressions.hpp"
#include "boost/log/expressions/keyword.hpp"

#include "boost/log/sources/logger.hpp"
#include "boost/log/sources/severity_feature.hpp"
#include "boost/log/sources/severity_logger.hpp"
#include "boost/log/sources/record_ostream.hpp" 
#include "boost/log/sources/global_logger_storage.hpp"

#include "boost/log/sinks/basic_sink_frontend.hpp"
#include "boost/log/sinks/basic_sink_backend.hpp"
#include "boost/log/sinks/text_ostream_backend.hpp"

#include "boost/log/utility/setup/file.hpp"
#include "boost/log/utility/setup/common_attributes.hpp"

#include "boost/log/attributes/scoped_attribute.hpp" 
#include "boost/log/attributes/current_process_id.hpp"
#include "boost/log/attributes/current_process_name.hpp"
#include "boost/log/attributes/current_thread_id.hpp"
#include "boost/log/attributes/function.hpp"
#include "boost/log/attributes/named_scope.hpp"

#include "boost/log/support/date_time.hpp"

#include "boost/log/trivial.hpp"


MED_IMG_BEGIN_NAMESPACE

//BOOST_LOG_ATTRIBUTE_KEYWORD(line_id, "LineID", unsigned int)
//BOOST_LOG_ATTRIBUTE_KEYWORD(severity, "Severity", SeverityLevel)
//BOOST_LOG_ATTRIBUTE_KEYWORD(scope, "Scope", attrs::named_scope::value_type)
//BOOST_LOG_ATTRIBUTE_KEYWORD(timeline, "TimeStamp", boost::posix_time::ptime)

boost::mutex Logger::_s_mutex;

Logger* Logger::_s_instance = nullptr;

Logger::Logger() {
    init_i();
}

Logger::~Logger() {

}

Logger* Logger::instance() {
    if (!_s_instance) {
        boost::unique_lock<boost::mutex> locker(_s_mutex);

        if (!_s_instance) {
            _s_instance = new Logger();
        }
    }

    return _s_instance;
}

void MED_IMG_NAMESPACE::Logger::init_i() {

    _console_filer_level = TRACE;
    _file_filer_level = TRACE;

   _console_sink = boost::make_shared<ConsoleSinkType>();
   boost::shared_ptr<std::ostream> stream(&std::clog , boost::null_deleter());
   _console_sink->locked_backend()->add_stream(stream);
   /*_console_sink->set_formatter(
       expr::stream
       << std::hex << std::setw(8) << std::setfill('0')
       << "[" << expr::format_date_time< boost::posix_time::ptime >("TimeStamp", "%Y-%m-%d %H:%M:%S") << "]"
       << "<" << logging::trivial::severity << ">"
       << "(" << expr::attr<boost::log::attributes::current_thread_id::value_type >("ThreadID") << ")"
       << ": " << expr::smessage);*/

   _console_sink->set_formatter(
       expr::format("[%1%]<%2%>(%3%): %4%")  
       % expr::format_date_time< boost::posix_time::ptime >("TimeStamp", "%Y-%m-%d %H:%M:%S")  
       % logging::trivial::severity  
       % expr::attr<boost::log::attributes::current_thread_id::value_type >("ThreadID")  
       % expr::smessage );

   logging::core::get()->add_sink(_console_sink);
   _console_sink->set_filter(logging::trivial::severity >= _console_filer_level);

   //_file_sink = logging::add_file_log(
   //    keywords::file_name = "mi-%Y-%m-%d_%H-%M-%S.%N.log",
   //    keywords::target = "log",
   //    keywords::rotation_size = 10*1024*1024,
   //    keywords::time_based_rotation = sinks::file::rotation_at_time_point(0,0,0),
   //    keywords::format = "[%TimeStamp%]<Severity>" ,
   //    keywords::min_free_space = 300*1024*1024,
   //    keywords::max_size = 50*1024*1024);
   //logging::core::get()->add_sink(_file_sink);
   //_file_sink->set_filter(logging::trivial::severity >= _file_filer_level);

   //global attributes
   //logging::add_common_attributes();
   //logging::core::get()->add_global_attribute("Scope" , attrs::named_scope());
   //logging::core::get()->add_global_attribute("ProcessID" , attrs::current_process_id());
   //logging::core::get()->add_global_attribute("Process" , attrs::current_process_name());
   //logging::core::get()->add_global_attribute("ThreadID" , attrs::current_thread_id());


    //logging::core::get()->set_filter
}


MED_IMG_END_NAMESPACE

    

