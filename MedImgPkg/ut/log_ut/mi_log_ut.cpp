#include <cstddef>
#include <string>
#include <ostream>
#include <fstream>
#include <iomanip>
#include <boost/smart_ptr/shared_ptr.hpp>
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

// We define our own severity levels
enum severity_level
{
    normal,
    notification,
    warning,
    error,
    critical
};

BOOST_LOG_INLINE_GLOBAL_LOGGER_DEFAULT(mi_logger,src::severity_logger< severity_level >)

BOOST_LOG_ATTRIBUTE_KEYWORD(line_id, "LineID", unsigned int)
BOOST_LOG_ATTRIBUTE_KEYWORD(severity, "Severity", severity_level)
BOOST_LOG_ATTRIBUTE_KEYWORD(tag_attr, "Module", std::string)
BOOST_LOG_ATTRIBUTE_KEYWORD(scope, "Scope", attrs::named_scope::value_type)
BOOST_LOG_ATTRIBUTE_KEYWORD(timeline, "Timeline", attrs::timer::value_type)

#define MI_LOG(sev) BOOST_LOG_SEV(mi_logger::get() , sev)

src::severity_logger< severity_level > io_lg;
#define MI_IO_LOG(sev) BOOST_LOG_SEV(io_lg , sev)

src::severity_logger< severity_level > util_lg;
#define MI_UTIL_LOG(sev) BOOST_LOG_SEV(util_lg , sev)


void logging_function()
{
    src::severity_logger< severity_level > slg;

    BOOST_LOG_SEV(slg, normal) << "A regular message";
    BOOST_LOG_SEV(slg, warning) << "Something bad is going on but I can handle it";
    BOOST_LOG_SEV(slg, critical) << "Everything crumbles, shoot me now!";
}

//[ example_tutorial_attributes_named_scope
void named_scope_logging()
{
    BOOST_LOG_NAMED_SCOPE("named_scope_logging");

    src::severity_logger< severity_level > slg;

    BOOST_LOG_SEV(slg, normal) << "Hello from the function named_scope_logging!";

    //mi_logger::get().add_attribute("Module", attrs::constant< std::string >("GLOBAL"));

    MI_LOG(normal) << "This is test nomal log.";

    MI_LOG(error) << "This is test MI log.";

    MI_IO_LOG(error) << "IO module log.";

    MI_UTIL_LOG(critical) << "Util module log";

    for (int i= 0 ; i< 2000 ; ++i)
    {
        MI_UTIL_LOG(critical) << "test rotation.";
    }
}
//]

//[ example_tutorial_attributes_tagged_logging
void tagged_logging()
{
    src::severity_logger< severity_level > slg;
    slg.add_attribute("Module", attrs::constant< std::string >("My tag value"));

    BOOST_LOG_SEV(slg, normal) << "Here goes the tagged record";
}
//]

//[ example_tutorial_attributes_timed_logging
void timed_logging()
{
    BOOST_LOG_SCOPED_THREAD_ATTR("Timeline", attrs::timer());

    src::severity_logger< severity_level > slg;
    BOOST_LOG_SEV(slg, normal) << "Starting to time nested functions";

    logging_function();

    BOOST_LOG_SEV(slg, normal) << "Stopping to time nested functions";
}
//]

// The operator puts a human-friendly representation of the severity level to the stream
std::ostream& operator<< (std::ostream& strm, severity_level level)
{
    static const char* strings[] =
    {
        "normal",
        "notification",
        "warning",
        "error",
        "critical"
    };

    if (static_cast< std::size_t >(level) < sizeof(strings) / sizeof(*strings))
        strm << strings[level];
    else
        strm << static_cast< int >(level);

    return strm;
}

void init()
{
    typedef sinks::synchronous_sink< sinks::text_ostream_backend > text_sink;
    boost::shared_ptr< text_sink > sink = boost::make_shared< text_sink >();

    sink->locked_backend()->add_stream(
        boost::make_shared< std::ofstream >("sample.log"));

    sink->set_formatter
        (
        expr::stream
        << std::hex << std::setw(8) << std::setfill('0') << line_id << std::dec << std::setfill(' ')
        << ": <" << severity << ">\t"
        << "(" << scope << ") "
        << expr::if_(expr::has_attr(tag_attr))
        [
            expr::stream << "[" << tag_attr << "] "
        ]
    << expr::if_(expr::has_attr(timeline))
        [
            expr::stream << "[" << timeline << "] "
        ]
    << expr::smessage
        );


    logging::core::get()->add_sink(sink);

    // Add attributes
    logging::add_common_attributes();
    logging::core::get()->add_global_attribute("Scope", attrs::named_scope());
}

void init2()
{
    //text file sink
    boost::shared_ptr< sinks::text_file_backend > file_sink_backend =
        boost::make_shared< sinks::text_file_backend >(
        keywords::file_name = "sign-%Y-%m-%d_%H-%M-%S.%5N.log", 
        keywords::enable_final_rotation = true,
        keywords::rotation_size = 1*1024*1024,
        keywords::time_based_rotation = sinks::file::rotation_at_time_point(0,0,0)
        );

    file_sink_backend->set_file_collector(sinks::file::make_collector(
        keywords::target = "logs",
        keywords::max_size = 16 * 1024 * 1024,
        keywords::min_free_space = 100 * 1024 * 1024,
        keywords::max_files = 512
        ));

    typedef sinks::synchronous_sink< sinks::text_file_backend > file_sink_t;
    boost::shared_ptr< file_sink_t > file_sink(new file_sink_t(file_sink_backend));
    file_sink->set_formatter
        (
        expr::stream
        << std::hex << std::setw(8) << std::setfill('0') << line_id << std::dec << std::setfill(' ')
        << ": <" << severity << ">\t"
        << "(" << scope << ") "
        << expr::if_(expr::has_attr(tag_attr))
        [
            expr::stream << "[" << tag_attr << "] "
        ]
    << expr::if_(expr::has_attr(timeline))
        [
            expr::stream << "[" << timeline << "] "
        ]
    << expr::smessage
        );


    //text stream sink
    boost::shared_ptr< sinks::text_ostream_backend > stream_sink_backend =
        boost::make_shared< sinks::text_ostream_backend >();
    stream_sink_backend->add_stream(
        boost::shared_ptr< std::ostream >(&std::clog, boost::null_deleter()));
    stream_sink_backend->auto_flush(true);
     typedef sinks::synchronous_sink< sinks::text_ostream_backend > stream_sink_t;
    boost::shared_ptr< stream_sink_t > stream_sink(new stream_sink_t(stream_sink_backend));
    stream_sink->set_formatter
        (
        expr::stream
        << std::hex << std::setw(8) << std::setfill('0') << line_id << std::dec << std::setfill(' ')
        << ": <" << severity << "> "
        << "(" << scope << ") "
        << expr::if_(expr::has_attr(tag_attr))
        [
            expr::stream << "[" << tag_attr << "] "
        ]
    << expr::if_(expr::has_attr(timeline))
        [
            expr::stream << "[" << timeline << "] "
        ]
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
    logging::core::get()->add_global_attribute("Module" , attrs::current_thread_id());


    mi_logger::get().add_attribute("Module", attrs::constant< std::string >("Global"));
    io_lg.add_attribute("Module" , attrs::constant<std::string>("IO"));
    util_lg.add_attribute("Module" , attrs::constant<std::string>("UTIL"));
}

void init1() 
{
    auto file_sink = logging::add_file_log(
        keywords::target = "log",
        keywords::file_name = "sign-%Y-%m-%d_%H-%M-%S.%N.log", 
        keywords::rotation_size = 10*1024,
        keywords::time_based_rotation = sinks::file::rotation_at_time_point(0,0,0),
        //keywords::format = "[%TimeStamp%] (%Severity%) <%LineID%> <%Scope%> <%ProcessID%> <%Process%> <%ThreadID%> {%Module%}: %Message%",
        keywords::min_free_space = 3*1024*1024 
        );

    file_sink->set_formatter
        (
        expr::stream
        << std::hex << std::setw(8) << std::setfill('0') << line_id << std::dec << std::setfill(' ')
        << ": <" << severity << ">\t"
        << "(" << scope << ") "
        << expr::if_(expr::has_attr(tag_attr))
        [
            expr::stream << "[" << tag_attr << "] "
        ]
    << expr::if_(expr::has_attr(timeline))
        [
            expr::stream << "[" << timeline << "] "
        ]
    << expr::smessage
        );

    logging::add_common_attributes();
    logging::core::get()->add_global_attribute("Scope" , attrs::named_scope());
    logging::core::get()->add_global_attribute("ProcessID" , attrs::current_process_id());
    logging::core::get()->add_global_attribute("Process" , attrs::current_process_name());
    logging::core::get()->add_global_attribute("ThreadID" , attrs::current_thread_id());
    logging::core::get()->add_global_attribute("Module" , attrs::current_thread_id());


    mi_logger::get().add_attribute("Module", attrs::constant< std::string >("Global"));
    io_lg.add_attribute("Module" , attrs::constant<std::string>("IO"));
    util_lg.add_attribute("Module" , attrs::constant<std::string>("UTIL"));

    //file_sink->set_filter(severity >= warning);
}

int main(int, char*[])
{
    init2();

    named_scope_logging();
    //tagged_logging();
    //timed_logging();

    return 0;
}