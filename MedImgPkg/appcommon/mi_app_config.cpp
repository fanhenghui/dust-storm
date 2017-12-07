#include "mi_app_config.h"
#include <fstream>
#include <sstream>

#include "util/mi_string_number_converter.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

AppConfig* AppConfig::_instance = nullptr;
boost::mutex AppConfig::_mutex;

void AppConfig::init_i() {
    _test_data_root = "";
    _db_ip = "127.0.0.1";
    _db_port = "3306";
    _db_user = "root";
    _db_pwd = "123456";
    _db_name = "med_img_db";
    _db_server_port = "8888";
    _cache_db_ip = "127.0.0.1";
    _cache_db_port = "3306";
    _cache_db_user = "root";
    _cache_db_pwd = "123456";
    _cache_db_name = "med_img_cache_db";
    _cache_db_path = "";
    _expected_fps = 30;
    _nodule_possibility_threshold = 0.97f;
    _pytorch_path = "";
    _py_interface_path = "";
    _pacs_server_ae_title = "DCM4CHEE";
    _pacs_server_host = "127.0.0.1";
    _pacs_server_port = 11112;
    _pacs_client_ae_title = "DBS";
    _pacs_client_port = 11115;
}

AppConfig::AppConfig() {
    init_i();

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
        if (line.empty()) {
            continue;
        }
        if (line[0] == '#') {
            continue;
        }
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
        } else if (tag == "DBServerPort"){
            _db_server_port = context;
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
        } else if (tag == "CacheDBPath") {
            _cache_db_path = context;
        } else if (tag == "PytorchPath") {
            _pytorch_path = context; 
        } else if (tag == "PyInterfacePath") {
            _py_interface_path = context;   
        } else if (tag == "PACSServerAETitle") {
            _pacs_server_ae_title = context;   
        } else if (tag == "PACSServerHost") {
            _pacs_server_host = context;   
        } else if (tag == "PACSServerPort") {
            StrNumConverter<unsigned short> conv;
            _pacs_server_port = conv.to_num(context);   
        } else if (tag == "PACSClientAETitle") {
            _pacs_client_ae_title = context;   
        } else if (tag == "PACSClientPort") {
            StrNumConverter<unsigned short> conv;
            _pacs_client_port = conv.to_num(context);   
        }
    }
}

AppConfig::~AppConfig() {}

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

void AppConfig::get_cache_db_info(std::string& ip_port, std::string& user, std::string& pwd, std::string& db_name, std::string& path) {
    ip_port = _cache_db_ip + ":" + _cache_db_port;
    user = _cache_db_user;
    pwd = _cache_db_pwd;
    db_name = _cache_db_name;
    path = _cache_db_path;
}

void AppConfig::get_db_info(std::string& ip_port, std::string& user, std::string& pwd, std::string& db_name) {
    ip_port = _db_ip  + ":" + _db_port;
    user = _db_user;
    pwd = _db_pwd; 
    db_name = _db_name;
}

void AppConfig::get_db_server_host(std::string& ip, std::string& port) {
    ip = _db_ip;
    port = _db_server_port;
}

int AppConfig::get_expected_fps() const {
    return _expected_fps;
}

float AppConfig::get_nodule_possibility_threshold() const {
    return _nodule_possibility_threshold;
}

std::string AppConfig::get_pytorch_path() const {
    return _pytorch_path;
}

std::string AppConfig::get_py_interface_path() const {
    return _py_interface_path;
}

void AppConfig::get_pacs_info(std::string& server_ae_title, std::string& server_host ,unsigned short& server_port,
    std::string& client_ae_title, unsigned short& client_port) {
    server_ae_title = _pacs_server_ae_title;
    server_host = _pacs_server_host;
    server_port = _pacs_server_port;
    client_ae_title = _pacs_client_ae_title;
    client_port = _pacs_client_port;
}

MED_IMG_END_NAMESPACE