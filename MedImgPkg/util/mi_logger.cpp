#include "mi_logger.h"

MED_IMG_BEGIN_NAMESPACE

boost::mutex Logger::_s_mutex;

Logger* Logger::_s_instance = nullptr;

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

MED_IMG_END_NAMESPACE

    

