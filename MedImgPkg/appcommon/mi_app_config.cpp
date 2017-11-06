#include "mi_app_config.h"
#include <fstream>
#include <sstream>

#include "util/mi_string_number_converter.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

AppConfig* AppConfig::_instance = nullptr;
boost::mutex AppConfig::_mutex;

AppConfig::AppConfig() {
    _expected_fps = 30;
    _nodule_possibility_threshold = 0.97f;
    std::ifstream in("../config/app_config");

    if (!in.is_open()) {
        MI_APPCOMMON_LOG(MI_ERROR) << "can't open configure file.";
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
        } else if (tag == "ExpectedFPS") {
            StrNumConverter<int> conv;
            _expected_fps = conv.to_num(context);
        } else if (tag == "PossibilityThreshold") {
            StrNumConverter<float> conv;
            _nodule_possibility_threshold = conv.to_num(context);
        } else if (tag == "RemoteDBIP") {
            _db_ip = context;
        } else if (tag == "RemoteDBPort") {
            _db_port = context;
        } else if (tag == "RemoteDBUser") {
            _db_user = context;
        } else if (tag == "RemoteDBPWD") {
            _db_pwd = context;
        } else if (tag == "RemoteDBName") {
            _db_name = context;
        } else if (tag == "CacheDBIP") {
            _cache_db_ip = context;
        } else if (tag == "CacheDBPort") {
            _cache_db_port = context;
        } else if (tag == "CacheDBUser") {
            _cache_db_user = context;
        } else if (tag == "CacheDBPWD") {
            _cache_db_pwd = context;
        } else if (tag == "CacheDBName") {
            _cache_db_name = context;
        } 
    }
}

AppConfig::~AppConfig() {

}

std::string AppConfig::get_config_root() const {
    return std::string("../config/");
}

std::string AppConfig::get_log_config_file() const {
    return this->get_config_root() + std::string("log_config");
}

AppConfig* AppConfig::instance() {
    if (nullptr == _instance) {
        boost::mutex::scoped_lock locker(_mutex);
        if (nullptr == _instance) {
            _instance = new AppConfig();
        }
    }
    return _instance;
}

std::string AppConfig::get_test_data_root() const {
    return _test_data_root;
}

void AppConfig::get_cache_db_info(std::string& ip_port, std::string& user, std::string& pwd, std::string& db_name) {
    ip_port = _cache_db_ip + ":" + _cache_db_port;
    user = _cache_db_user;
    pwd = _cache_db_pwd;
    db_name = _cache_db_name;
}

void AppConfig::get_db_info(std::string& ip_port, std::string& user, std::string& pwd, std::string& db_name) {
    ip_port = _db_ip  + ":" + _db_port;
    user = _db_user;
    pwd = _db_pwd; 
    db_name = _db_name;
}

int AppConfig::get_expected_fps() const {
    return _expected_fps;
}

float AppConfig::get_nodule_possibility_threshold() const {
    return _nodule_possibility_threshold;
}

MED_IMG_END_NAMESPACE