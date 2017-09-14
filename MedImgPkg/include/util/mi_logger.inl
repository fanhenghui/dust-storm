boost::mutex Logger::_s_mutex;
Logger* Logger::_s_instance = nullptr;

typedef sinks::synchronous_sink< sinks::text_file_backend > file_sink_t;
typedef sinks::synchronous_sink< sinks::text_ostream_backend > stream_sink_t;

std::ostream& operator<< (std::ostream& strm, SeverityLevel lvl)
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

struct Logger::InnerSink {
    //boost::weak_ptr<file_sink_t> file_sink;
    //boost::weak_ptr<stream_sink_t> stream_sink;
};

Logger::Logger() : _inner_sink(new InnerSink)
    , _stream_filer_level(DEBUG)
    , _file_filer_level(DEBUG)
    , _target_dir("log")
    , _file_name_format("sign-%Y-%m-%d_%H-%M-%S.%N.log")
    , _rotation_size(10*1024)
    , _min_free_space(3*1024*1024)
    , _time_based_rotation(TimeHMS(0,0,0))
{
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

void Logger::initialize() {
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
        << expr::if_(expr::has_attr(module))
        [
            expr::stream << "[" << module << "] "
        ]
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
        << expr::if_(expr::has_attr(module))
        [
            expr::stream << "[" << module << "] "
        ]
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
    logging::core::get()->add_global_attribute("Module" , attrs::current_thread_id());

    this->register_logger_module(G_UTIL_LG , "Util");
} 

void Logger::register_logger_module(src::severity_logger<SeverityLevel>& lg, const std::string& module_name) {
    lg.add_attribute("Module" , attrs::constant<std::string>(module_name));
}

void Logger::finalize() {
    if (_s_instance) {
        boost::unique_lock<boost::mutex> locker(_s_mutex);
        delete _s_instance;
        _s_instance = nullptr;
    }
}

