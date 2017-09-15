#include "log/mi_logger.h"

#include "boost/smart_ptr/shared_ptr.hpp"
#include "boost/smart_ptr/weak_ptr.hpp"
#include "boost/smart_ptr/make_shared_object.hpp"

#include "boost/log/sinks/sync_frontend.hpp"
#include "boost/log/sinks/async_frontend.hpp"
#include "boost/log/sinks/text_ostream_backend.hpp"
#include "boost/log/attributes/scoped_attribute.hpp"
#include "boost/log/utility/setup/common_attributes.hpp"
#include "boost/log/sources/global_logger_storage.hpp"
#include "boost/log/utility/setup/file.hpp"
#include "boost/core/null_deleter.hpp"

typedef sinks::synchronous_sink< sinks::text_file_backend > file_sink_t;
typedef sinks::synchronous_sink< sinks::text_ostream_backend > stream_sink_t;

namespace medical_imaging {
Logger* Logger::_s_instance = nullptr;
boost::mutex Logger::_s_mutex;

Logger::Logger() : //_inner_sink(new InnerSink)
_stream_filer_level(MI_DEBUG)
    , _file_filer_level(MI_DEBUG)
    , _target_dir("log")
    , _file_name_format("sign-%Y-%m-%d_%H-%M-%S.%N.log")
    , _rotation_size(10*1024)
    , _min_free_space(3*1024*1024)
    , _time_based_rotation(TimeHMS(0,0,0))
{
}

Logger::~Logger() {

}

void Logger::initialize()
{
    boost::shared_ptr< sinks::text_file_backend > file_sink_backend =
        boost::make_shared< sinks::text_file_backend >(
        keywords::file_name = "mi-%Y-%m-%d_%H-%M-%S.%5N.log", 
        //keywords::enable_final_rotation = true,
        keywords::rotation_size = 1 * 1024 * 1024,
        keywords::time_based_rotation = sinks::file::rotation_at_time_point(0,0,0)
        );

    file_sink_backend->set_file_collector(sinks::file::make_collector(
        keywords::target = "logs",
        keywords::max_size = 16 * 1024 * 1024,
        keywords::min_free_space = 100 * 1024 * 1024
        //keywords::max_files = 512
        ));

    boost::shared_ptr< file_sink_t > file_sink(new file_sink_t(file_sink_backend));
    //_inner_sink->file_sink = file_sink;
    file_sink->set_formatter
        (
        expr::stream
        << std::hex << std::setw(8) << std::setfill('0') << line_id << std::dec << std::setfill(' ')
        << ": <" << severity << "> "
        //<< "(" << scope << ") "
        << expr::if_(expr::has_attr(timeline))
        [
            expr::stream << "[" << timeline << "] "
        ]
    << "{" << thread_id << "} "
        << expr::smessage
        );


    //text stream sink
    boost::shared_ptr< sinks::text_ostream_backend > stream_sink_backend =
        boost::make_shared< sinks::text_ostream_backend >();
    stream_sink_backend->add_stream(
        boost::shared_ptr< std::ostream >(&std::clog, boost::null_deleter()));
    stream_sink_backend->auto_flush(true);
    boost::shared_ptr< stream_sink_t > stream_sink(new stream_sink_t(stream_sink_backend));
    //_inner_sink->stream_sink= stream_sink;
    stream_sink->set_formatter
        (
        expr::stream
        << std::hex << std::setw(8) << std::setfill('0') << line_id << std::dec << std::setfill(' ')
        << ": <" << severity << "> "
        //<< "(" << scope << ") "
        << expr::if_(expr::has_attr(timeline))
        [
            expr::stream << "[" << timeline << "] "
        ]
    << "{" << thread_id << "} "
        << expr::smessage
        );


    boost::shared_ptr< logging::core > core = logging::core::get();
    core->add_sink(file_sink);
    core->add_sink(stream_sink);

    logging::add_common_attributes();
    logging::core::get()->add_global_attribute("Scope" , attrs::named_scope());
    logging::core::get()->add_global_attribute("ProcessID" , attrs::current_process_id());
    logging::core::get()->add_global_attribute("Process" , attrs::current_process_name());
    logging::core::get()->add_global_attribute("ThreadID" , attrs::current_thread_id());
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

}