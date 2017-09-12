#include <stdio.h>
#include <string>
#include <fstream>

#include "boost/shared_ptr.hpp"  
#include "boost/make_shared.hpp"
#include "boost/core/null_deleter.hpp"

#include "boost/log/core.hpp"
#include "boost/log/trivial.hpp"
#include "boost/log/expressions.hpp"
#include "boost/log/utility/setup/file.hpp"
#include "boost/log/utility/setup/common_attributes.hpp"
#include "boost/log/sources/severity_feature.hpp"
#include "boost/log/sources/severity_logger.hpp"
#include "boost/log/sources/record_ostream.hpp" 
#include "boost/log/sinks/basic_sink_frontend.hpp"
#include "boost/log/sinks/basic_sink_backend.hpp"
#include "boost/log/sinks/text_ostream_backend.hpp"

#include "boost/log/support/date_time.hpp"
#include "boost/log/attributes/scoped_attribute.hpp" 

#include "boost/thread.hpp"

namespace logging = boost::log;
namespace src = boost::log::sources;
namespace sinks = boost::log::sinks;
namespace expr = boost::log::expressions;  
namespace keywords = boost::log::keywords;

void init() {
    logging::core::get()->set_filter(logging::trivial::severity >= logging::trivial::info);
    logging::register_simple_formatter_factory<logging::trivial::severity_level , char>("Severity");

    //construct text sinks
    typedef sinks::synchronous_sink<sinks::text_ostream_backend> TextSink;
    boost::shared_ptr<TextSink> text_sink = boost::make_shared<TextSink>();
    //Add as stream to write log to
    //text_sink->locked_backend()->add_stream(boost::make_shared<std::ofstream>("sign.log"));
    boost::shared_ptr<std::ostream> stream(&std::clog , boost::null_deleter());
    text_sink->locked_backend()->add_stream(stream);
    text_sink->set_formatter(
        expr::format("\033[31m [%1%]<%2%>(%3%): %4%")  
        % expr::format_date_time< boost::posix_time::ptime >("TimeStamp", "%Y-%m-%d %H:%M:%S")  
        % logging::trivial::severity  
        % expr::attr<boost::log::attributes::current_thread_id::value_type >("ThreadID")  
        % expr::smessage );
    logging::core::get()->add_sink(text_sink);

    logging::add_file_log(
        keywords::file_name = "sign-%Y-%m-%d_%H-%M-%S.%N.log", 
        keywords::rotation_size = 10*1024,
        keywords::time_based_rotation = sinks::file::rotation_at_time_point(0,0,0),
        keywords::format = "[%TimeStamp%] (%Severity%) : %Message%",
        keywords::min_free_space = 3*1024*1024
        );
    logging::add_common_attributes();
    src::severity_logger<logging::trivial::severity_level> lg;
    BOOST_LOG_SEV(lg , logging::trivial::info) << "thread id : " << boost::this_thread::get_id() << " Initialization succeeded.";
}

int main(int argc, char* argv[]) {
    init();
    BOOST_LOG_TRIVIAL(trace) << "a trace serverity message.";
    BOOST_LOG_TRIVIAL(debug) << "a debug serverity message.";
    BOOST_LOG_TRIVIAL(info) << "a info serverity message.";
    BOOST_LOG_TRIVIAL(warning) << "a warning serverity message.";
    BOOST_LOG_TRIVIAL(error) << "a error serverity message.";
    BOOST_LOG_TRIVIAL(fatal) << "a fatal serverity message.";

    return 0;
}
