#include "log/mi_logger.h"

#ifdef WIN32
#include <Windows.h>
#endif

#include <fstream>

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

namespace {
void stream_formatter(logging::record_view const& rec, logging::formatting_ostream& strm)
{
    using namespace medical_imaging;
    auto lvl = logging::extract< SeverityLevel >("Severity", rec);
    std::string sev = "";
    
#ifdef WIN32
    if(lvl) {
        if (static_cast<int>(lvl.get()) < LEVEL_NUM ) {
            sev = LEVEL_STR[lvl.get()];
        }
        HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
        switch(lvl.get()) {
            case MI_TRACE :
                SetConsoleTextAttribute(handle , FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
                break;
            case MI_DEBUG :
                SetConsoleTextAttribute(handle , FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN);
                break;
            case MI_INFO :
                SetConsoleTextAttribute(handle , FOREGROUND_INTENSITY | FOREGROUND_GREEN | FOREGROUND_BLUE);
                break;
            case MI_WARNING :
                SetConsoleTextAttribute(handle , FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_BLUE);
                break;
            case MI_ERROR :
                SetConsoleTextAttribute(handle , FOREGROUND_INTENSITY | FOREGROUND_RED);
                break;
            case MI_FATAL :
                SetConsoleTextAttribute(handle , FOREGROUND_RED);
                break;
            default:
                SetConsoleTextAttribute(handle , FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
        }
    }
#else
    if(lvl) {
        if (static_cast<int>(lvl.get()) < LEVEL_NUM ) {
            sev = LEVEL_STR[lvl.get()];
        }
        switch(lvl.get()) {
            case MI_TRACE :
                strm << "\033[37m";
                break;
            case MI_DEBUG :
                strm << "\033[33m";
                break;
            case MI_INFO :
                strm << "\033[36m";
                break;
            case MI_WARNING :
                strm << "\033[35m";
                break;
            case MI_ERROR :
                strm << "\033[31m";
                break;
            case MI_FATAL :
                strm << "\033[31m";
                break;
            default:
                strm << "\033[37m";
        }
    }
#endif

    strm << std::hex << std::setw(8) << std::setfill('0') << 
        logging::extract< unsigned int >("LineID", rec) << std::dec << std::setfill(' ')
        << ": <" << sev << "> "
        << "{" << logging::extract< boost::log::aux::thread::id >("ThreadID", rec) << "} "
        << rec[expr::smessage];

#ifndef WIN32
    if(lvl) {
        strm << "\033[1m";  
    }
#endif
}

void file_formatter(logging::record_view const& rec, logging::formatting_ostream& strm)
{
    using namespace medical_imaging;
    auto lvl = logging::extract< SeverityLevel >("Severity", rec);

    std::string sev = "";
    if(lvl) {
        if (static_cast<int>(lvl.get()) < LEVEL_NUM ) {
            sev = LEVEL_STR[lvl.get()];
        }
    }
    strm << std::hex << std::setw(8) << std::setfill('0') << 
        logging::extract< unsigned int >("LineID", rec) << std::dec << std::setfill(' ')
        << ": <" << sev << "> "
        << "{" << logging::extract< boost::log::aux::thread::id >("ThreadID", rec) << "} "
        << rec[expr::smessage];
}
}

MED_IMG_BEGIN_NAMESPACE

Logger* Logger::_s_instance = nullptr;
boost::mutex Logger::_s_mutex;

Logger::Logger() : //_inner_sink(new InnerSink)
    _init(false)
    , _stream_filer_level(MI_DEBUG)
    , _file_filer_level(MI_DEBUG)
    , _file_target_dir("logs")
    , _file_name_format("mi-%Y-%m-%d_%H-%M-%S.%5N.log")
    , _min_free_space(1024*1024*1024)
    , _max_file_size(32*10241*1024)
    , _rotation_size(4*1024*1024)
    , _rotation_time(TimeHMS(0,0,0))
    , _config_file("") {
}

Logger::~Logger() {
}

void Logger::initialize()
{
    if (_init) {
        return;
    }

    read_config_file();

    boost::shared_ptr< sinks::text_file_backend > file_sink_backend =
        boost::make_shared< sinks::text_file_backend >(
        keywords::file_name = _file_name_format, 
        //keywords::enable_final_rotation = true,//version 1.65
        keywords::rotation_size = _rotation_size,
        keywords::time_based_rotation = sinks::file::rotation_at_time_point(0,0,0)
        );

    file_sink_backend->set_file_collector(sinks::file::make_collector(
        keywords::target = _file_target_dir,
        keywords::max_size = _max_file_size,
        keywords::min_free_space = _min_free_space
        //keywords::max_files = 512//version 1.65
        ));

    boost::shared_ptr< file_sink_t > file_sink(new file_sink_t(file_sink_backend));
    //_inner_sink->file_sink = file_sink;
    ////set format use expression
    //file_sink->set_formatter
    //    (
    //    expr::stream
    //    << std::hex << std::setw(8) << std::setfill('0') << line_id << std::dec << std::setfill(' ')
    //    << ": <" << severity << "> "
    //    //<< "(" << scope << ") "
    //    << expr::if_(expr::has_attr(timeline))
    //    [
    //        expr::stream << "[" << timeline << "] "
    //    ]
    //<< "{" << thread_id << "} "
    //    << expr::smessage
    //    );
    ////set format use function
    file_sink->set_formatter(&file_formatter);


    //text stream sink
    boost::shared_ptr< sinks::text_ostream_backend > stream_sink_backend =
        boost::make_shared< sinks::text_ostream_backend >();
    stream_sink_backend->add_stream(
        boost::shared_ptr< std::ostream >(&std::clog, boost::null_deleter()));
    stream_sink_backend->auto_flush(true);
    boost::shared_ptr< stream_sink_t > stream_sink(new stream_sink_t(stream_sink_backend));
    //_inner_sink->stream_sink= stream_sink;

    ////set format use expression
    //stream_sink->set_formatter
    //    (
    //    expr::stream
    //    << std::hex << std::setw(8) << std::setfill('0') << line_id << std::dec << std::setfill(' ')
    //    << ": <" << severity << "> "
    //    //<< "(" << scope << ") "
    //    << expr::if_(expr::has_attr(timeline))
    //    [
    //        expr::stream << "[" << timeline << "] "
    //    ]
    //<< "{" << thread_id << "} "
    //    << expr::smessage
    //    );
    ////set format use function
    stream_sink->set_formatter(&stream_formatter);

    stream_sink->set_filter(severity >= _stream_filer_level);
    file_sink->set_filter(severity >= _file_filer_level );

    boost::shared_ptr< logging::core > core = logging::core::get();
    core->add_sink(file_sink);
    core->add_sink(stream_sink);

    logging::add_common_attributes();
    logging::core::get()->add_global_attribute("Scope" , attrs::named_scope());
    logging::core::get()->add_global_attribute("ProcessID" , attrs::current_process_id());
    logging::core::get()->add_global_attribute("Process" , attrs::current_process_name());
    logging::core::get()->add_global_attribute("ThreadID" , attrs::current_thread_id());

    _init = true;
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

void Logger::bind_config_file(const std::string& file_name) {
    _config_file = file_name;
}

void Logger::set_file_direction(const std::string& tag_dir) {
    _file_target_dir = tag_dir;
}

void Logger::set_file_name_format(const std::string& name_format) {
    _file_name_format = name_format;
}

void Logger::set_file_max_size(unsigned int size) {
    _max_file_size = size;
}

void Logger::set_file_rotation_size(unsigned int size) {
    _rotation_size = size;
}

void Logger::set_file_rotation_time(unsigned char hour , unsigned char min , unsigned char sec) {
    _rotation_time = TimeHMS(hour, min, sec);
}

void Logger::set_min_free_space(unsigned int free_space) {
    _min_free_space = free_space;
}

void Logger::filter_level_stream(SeverityLevel level) {
    _stream_filer_level = level;
}

void Logger::filter_level_file(SeverityLevel level) {
    _file_filer_level = level;
}

void Logger::read_config_file() {
    if (_config_file.empty()) {
        return;
    }
    std::ifstream input_file(_config_file.c_str(), std::ios::in);
    if (!input_file.is_open()) {
        return;
    }
    std::string line;
    std::string tag;
    std::string equal;
    std::string context;
    while(std::getline(input_file,line)) {
        std::stringstream ss(line);
        ss >> tag >> equal >> context;
        if (tag == std::string("Level")) {
            static const int lvlnum = 6; 
            static const char* const lvlstr[lvlnum] = {
                "Trace",
                "Debug",
                "Info",
                "Warning",
                "Error",
                "Fatal",
            };
            for (int i = 0; i < lvlnum ; ++i) {
                if(context == lvlstr[i]) {
                    _file_filer_level = static_cast<SeverityLevel>(i);
                    _stream_filer_level = static_cast<SeverityLevel>(i);
                    break;
                }
            }
        } else if (tag == "MinFeeeSpace") {
            _min_free_space = atoi(context.c_str())*1024*1204;
        } else if (tag == "MaxLogSize") {
            _max_file_size = atoi(context.c_str())*1024*1024;
        } else if (tag == "RotationSize") {
            _rotation_size = atoi(context.c_str())*1024*1024;
        } else if (tag == "TargetDirection") {
            _file_target_dir = context;
        }
    }
    input_file.close();
}

MED_IMG_END_NAMESPACE