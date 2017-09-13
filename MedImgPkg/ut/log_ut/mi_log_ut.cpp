#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>

#include "boost/shared_ptr.hpp"  
#include "boost/make_shared.hpp"
#include "boost/core/null_deleter.hpp"

#include "boost/log/core.hpp"
#include "boost/log/trivial.hpp"
#include "boost/log/expressions.hpp"
#include "boost/log/expressions/keyword.hpp"
#include "boost/log/utility/setup/file.hpp"
#include "boost/log/utility/setup/console.hpp"
#include "boost/log/utility/setup/common_attributes.hpp"
#include "boost/log/sources/logger.hpp"
#include "boost/log/sources/severity_feature.hpp"
#include "boost/log/sources/severity_logger.hpp"
#include "boost/log/sources/record_ostream.hpp" 
#include "boost/log/sources/global_logger_storage.hpp"
#include "boost/log/sinks/basic_sink_frontend.hpp"
#include "boost/log/sinks/basic_sink_backend.hpp"
#include "boost/log/sinks/text_ostream_backend.hpp"

#include "boost/log/support/date_time.hpp"
#include "boost/log/attributes/scoped_attribute.hpp" 
#include "boost/log/attributes/current_process_id.hpp"
#include "boost/log/attributes/current_process_name.hpp"
#include "boost/log/attributes/current_thread_id.hpp"
#include "boost/log/attributes/function.hpp"
#include "boost/log/attributes/named_scope.hpp"

#include "boost/thread.hpp"

namespace logging = boost::log;
namespace src = boost::log::sources;
namespace sinks = boost::log::sinks;
namespace expr = boost::log::expressions;  
namespace keywords = boost::log::keywords;
namespace attrs = boost::log::attributes;

class current_module_name : public attrs::constant<std::string>
{
    typedef constant< std::string> base_type;
public:
    current_module_name() : base_type("io") {}
    explicit current_module_name(attrs::cast_source const& source) : base_type(source) {}
};

//std::string current_module_name()
//{
//    return std::string("io");
//}

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
        expr::format("[%1%]<%2%>(%3%): %4%")  
        % expr::format_date_time< boost::posix_time::ptime >("TimeStamp", "%Y-%m-%d %H:%M:%S")  
        % logging::trivial::severity  
        % expr::attr<boost::log::attributes::current_thread_id::value_type >("ThreadID")  
        % expr::smessage );
    logging::core::get()->add_sink(text_sink);
    
    auto file_sink = logging::add_file_log(
        keywords::file_name = "sign-%Y-%m-%d_%H-%M-%S.%N.log", 
        keywords::rotation_size = 10*1024,
        keywords::time_based_rotation = sinks::file::rotation_at_time_point(0,0,0),
        keywords::format = "[%TimeStamp%] (%Severity%) <%LineID%> <%Scope%> <%ProcessID%> <%Process%> <%ThreadID%> {%ModouleName%}: %Message%",
        keywords::min_free_space = 3*1024*1024 
        );

    //add attributes
    logging::add_common_attributes();
    logging::core::get()->add_global_attribute("Scope" , attrs::named_scope());
    logging::core::get()->add_global_attribute("ProcessID" , attrs::current_process_id());
    logging::core::get()->add_global_attribute("Process" , attrs::current_process_name());
    logging::core::get()->add_global_attribute("ThreadID" , attrs::current_thread_id());
    logging::core::get()->add_global_attribute("ModouleName" , current_module_name());
    
    src::severity_logger<logging::trivial::severity_level> lg;
    BOOST_LOG_SEV(lg , logging::trivial::info) << "thread id : " << boost::this_thread::get_id() << " Initialization succeeded.";
}

void func0() {
    //;
    //BOOST_LOG_FUNCTION();
    BOOST_LOG_NAMED_SCOPE("func0");
    //BOOST_LOG_FUNCTION();
    src::severity_logger<logging::trivial::severity_level> lg; 
    BOOST_LOG_SEV(lg , logging::trivial::error) << "in func 0";

    //BOOST_LOG_FUNCTION();
    BOOST_LOG_TRIVIAL(info) << "in func 0";
}

void func1() {
    //BOOST_LOG_FUNCTION();
    //BOOST_LOG_FUNCTION();
    BOOST_LOG_NAMED_SCOPE("func1");
    //BOOST_LOG_FUNCTION();
    src::severity_logger<logging::trivial::severity_level> lg;
    BOOST_LOG_SEV(lg , logging::trivial::fatal) << "in func 1";
}


int main(int argc, char* argv[]) {
    std::cout << current_module_name() << std::endl;
    init();

    //BOOST_LOG_FUNCTION();
    
    func0();
    func1();
    
    BOOST_LOG_TRIVIAL(trace) << "a trace serverity message.";
    BOOST_LOG_TRIVIAL(debug) << "a debug serverity message.";
    BOOST_LOG_TRIVIAL(info) << "a info serverity message.";
    BOOST_LOG_TRIVIAL(warning) << "a warning serverity message.";
    BOOST_LOG_TRIVIAL(error) << "a error serverity message.";
    BOOST_LOG_TRIVIAL(fatal) << "a fatal serverity message.";

    return 0;
}


//enum severity_level {
//    normal,
//    notification,
//    warning,
//    error,
//    critical,
//};
//
//void logging_function1() {
//     src::severity_logger< severity_level > lg;
//    logging::record rec = lg.open_record(keywords::severity = error);
//    if(rec) {
//        logging::record_ostream strm(rec);
//        strm << "hello world.";
//        strm.flush();
//        lg.push_record(boost::move(rec));
//    }
//}
//
//
//
//void logging_function2() {
//    src::severity_logger<severity_level> lg;
//    lg.add_attribute("Tag", attrs::constant< std::string >("My tag value"));
//    BOOST_LOG_SEV(lg , normal) << "hello world2.";
//}
//
//int main(int argc , char* argv[]) {
//    logging::add_common_attributes();
//    logging_function1();
//    logging_function2();
//    return 0;
//}