#include "mi_model_anonymization.h"

MED_IMG_BEGIN_NAMESPACE

ModelAnonymization::ModelAnonymization():_annoymization_flag(false) {

}

ModelAnonymization::~ModelAnonymization() {

}

void ModelAnonymization::set_anonymization_flag(bool flag) {
    boost::mutex::scoped_lock locker(_mutex);
    _annoymization_flag = flag;
}

bool ModelAnonymization::get_anonymization_flag() const {
    boost::mutex::scoped_lock locker(_mutex);
    return _annoymization_flag;
}

MED_IMG_END_NAMESPACE