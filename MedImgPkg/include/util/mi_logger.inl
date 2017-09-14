
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

struct Logger::InnerSink
{
    boost::weak_ptr<file_sink_t> file_sink;
    boost::weak_ptr<stream_sink_t> stream_sink;
};

void Logger::init() {
    boost::shared_ptr< sinks::text_file_backend > file_sink_backend =
        boost::make_shared< sinks::text_file_backend >(
        keywords::file_name = "mi-%Y-%m-%d_%H-%M-%S.%5N.log", 
        keywords::enable_final_rotation = true,
        keywords::rotation_size = 1 * 1024 * 1024,
        keywords::time_based_rotation = sinks::file::rotation_at_time_point(0,0,0)
        );

    file_sink_backend->set_file_collector(sinks::file::make_collector(
        keywords::target = "logs",
        keywords::max_size = 16 * 1024 * 1024,
        keywords::min_free_space = 100 * 1024 * 1024,
        keywords::max_files = 512
        ));

    boost::shared_ptr< file_sink_t > file_sink(new file_sink_t(file_sink_backend));
    _inner_sink->file_sink = file_sink;
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
    boost::shared_ptr< stream_sink_t > stream_sink(new stream_sink_t(stream_sink_backend));
    _inner_sink->stream_sink= stream_sink;
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
} 

