#include "mi_configuration.h"

MED_IMG_BEGIN_NAMESPACE

boost::mutex Configuration::_s_mutex;

Configuration* Configuration::_s_instance = nullptr;

Configuration* Configuration::instance() {
    if (!_s_instance) {
        boost::unique_lock<boost::mutex> locker(_s_mutex);

        if (!_s_instance) {
            _s_instance = new Configuration();
        }
    }

    return _s_instance;
}

Configuration::~Configuration() {

}

ProcessingUnitType Configuration::get_processing_unit_type() {
    return _processing_unit_type;
}

Configuration::Configuration(): _processing_unit_type(CPU) {
    //TODO Check hardware processing unit . Check if has GPU
}

void Configuration::set_processing_unit_type(ProcessingUnitType type) {
    _processing_unit_type = type;
}

MED_IMG_END_NAMESPACE