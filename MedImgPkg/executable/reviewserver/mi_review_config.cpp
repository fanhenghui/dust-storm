#include "mi_review_config.h"
#include <fstream>
#include <sstream>

#include "util/mi_string_number_converter.h"
#include "mi_review_logger.h"

MED_IMG_BEGIN_NAMESPACE

ReviewConfig* ReviewConfig::_instance = nullptr;
boost::mutex ReviewConfig::_mutex;

ReviewConfig::ReviewConfig() {
    _expected_fps = 30;
    _nodule_p_th = 0.97f;
    std::ifstream in("../config/configure.txt");

    if (!in.is_open()) {
        MI_REVIEW_LOG(MI_ERROR) << "can't open configure file.";
        return;
    }

    std::string line;
    std::string tag;
    std::string equal;
    std::string context;
    while (std::getline(in , line)) {
        std::stringstream ss(line);
        ss >> tag >> equal >> context;

        if (tag == "TestDataRoot") {
            _test_data_root = context;
        } else if (tag == "DBPWD") {
            _db_wpd = context;
        } else if (tag == "ExpectedFPS") {
            StrNumConverter<int> conv;
            _expected_fps = conv.to_num(context);
        } else if (tag == "PossibilityThreshold") {
            StrNumConverter<float> conv;
            _nodule_p_th = conv.to_num(context);
        }
    }
}

ReviewConfig::~ReviewConfig() {

}

std::string ReviewConfig::get_config_root() const {
    return std::string("../config/");
}

std::string ReviewConfig::get_log_config_file() const {
    return this->get_config_root() + std::string("log_config");
}

ReviewConfig* ReviewConfig::instance() {
    if (nullptr == _instance) {
        boost::mutex::scoped_lock locker(_mutex);

        if (nullptr == _instance) {
            _instance = new ReviewConfig();
        }
    }

    return _instance;
}

std::string ReviewConfig::get_test_data_root() const {
    return _test_data_root;
}

std::string ReviewConfig::get_db_pwd() const {
    return _db_wpd;
}

int ReviewConfig::get_expected_fps() const {
    return _expected_fps;
}

float ReviewConfig::get_nodule_possibility_threshold() const {
    return _nodule_p_th;
}

MED_IMG_END_NAMESPACE