#include "mi_configure.h"
#include <fstream>
#include <sstream>

#include "util/mi_string_number_converter.h"
#include "mi_io_logger.h"

MED_IMG_BEGIN_NAMESPACE

Configure* Configure::_instance = nullptr;
boost::mutex Configure::_mutex;

void Configure::init() {
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
    _evaluation_limit = 20;
    _pytorch_path = "";
    _py_interface_path = "";
    _pacs_server_ae_title = "DCM4CHEE";
    _pacs_server_host = "127.0.0.1";
    _pacs_server_port = 11112;
    _pacs_client_ae_title = "DBS";
    _pacs_client_port = 11115;
    _processing_unit_type = GPU;
    _gpu_platform_type = CUDA;
}

void Configure::refresh() {
#ifdef WIN32
    std::ifstream in("./config/app_config");
#else
    std::ifstream in("../config/app_config");
#endif
    if (!in.is_open()) {
        MI_IO_LOG(MI_ERROR) << "can't open configure file.";
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
        } else if (tag == "ProbabilityThreshold") {
            StrNumConverter<float> conv;
            _nodule_possibility_threshold = conv.to_num(context);
        } else if (tag == "EvaluationLimit") {
            StrNumConverter<int> conv;
            _evaluation_limit = conv.to_num(context);
        } else if (tag == "DBIP") {
            _db_ip = context;
        } else if (tag == "DBPort") {
            _db_port = context;
        } else if (tag == "DBUser") {
            _db_user = context;
        } else if (tag == "DBPWD") {
            _db_pwd = context;
        } else if (tag == "DBName") {
            _db_name = context;
        } else if (tag == "DBPath") {
            _db_path = context;
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
        } else if (tag == "GPUPlatformType") {
            if (context == std::string("CUDA") || context == std::string("cuda") || context == std::string("Cuda")) {
                _gpu_platform_type = CUDA;
            } else if (context == std::string("OPENGL") || context == std::string("opengl") || context == std::string("OpenGL") ) {
                _gpu_platform_type = OPENGL;
            }
        }
    }
}

Configure::Configure() {
    init();
    refresh();
}

Configure::~Configure() {}

std::string Configure::get_config_root() const {
    return std::string("../config/");
}

std::string Configure::get_log_config_file() const {
    return this->get_config_root() + std::string("log_config");
}

Configure* Configure::instance() {
    if (nullptr == _instance) {
        boost::mutex::scoped_lock locker(_mutex);
        if (nullptr == _instance) {
            _instance = new Configure();
        }
    }
    return _instance;
}

std::string Configure::get_test_data_root() const {
    return _test_data_root;
}

void Configure::get_cache_db_info(std::string& ip_port, std::string& user, std::string& pwd, std::string& db_name, std::string& path) const {
    ip_port = _cache_db_ip + ":" + _cache_db_port;
    user = _cache_db_user;
    pwd = _cache_db_pwd;
    db_name = _cache_db_name;
    path = _cache_db_path;
}

void Configure::get_db_info(std::string& ip_port, std::string& user, std::string& pwd, std::string& db_name) const {
    ip_port = _db_ip  + ":" + _db_port;
    user = _db_user;
    pwd = _db_pwd; 
    db_name = _db_name;
}

std::string Configure::get_db_path() const {
    return _db_path;
}

void Configure::get_db_server_host(std::string& ip, std::string& port) const {
    ip = _db_ip;
    port = _db_server_port;
}

int Configure::get_expected_fps() const {
    return _expected_fps;
}

float Configure::get_evaluation_probability_threshold() const {
    return _nodule_possibility_threshold;
}

std::string Configure::get_pytorch_path() const {
    return _pytorch_path;
}

std::string Configure::get_py_interface_path() const {
    return _py_interface_path;
}

void Configure::get_pacs_info(std::string& server_ae_title, std::string& server_host ,unsigned short& server_port,
    std::string& client_ae_title, unsigned short& client_port) {
    server_ae_title = _pacs_server_ae_title;
    server_host = _pacs_server_host;
    server_port = _pacs_server_port;
    client_ae_title = _pacs_client_ae_title;
    client_port = _pacs_client_port;
}

ProcessingUnitType Configure::get_processing_unit_type() const {
    return _processing_unit_type;
}

void Configure::set_processing_unit_type(ProcessingUnitType type) {
    _processing_unit_type = type;
}

int Configure::get_evaluation_limit() const {
    return _evaluation_limit;
}

GPUPlatformType Configure::get_gpu_platform_type() const {
    return _gpu_platform_type;
}

Version Configure::get_evaluation_version(EvaluationType type) const {
    Version v;
#ifdef WIN32
    std::ifstream in("./config/app_config");
#else
    std::ifstream in("../config/app_config");
#endif
    
    if (!in.is_open()) {
        MI_IO_LOG(MI_ERROR) << "can't open configure file when get evaluation version.";
        return v;
    }

    std::string line;
    std::string tag;
    std::string equal;
    std::string context;

    const static std::map<EvaluationType, std::string>::value_type mapper_value[] = {
        std::map<EvaluationType, std::string>::value_type(LUNG_NODULE, "EvaLungNoduleVersion")
    };

    const static std::map<EvaluationType, std::string> mapper(mapper_value, mapper_value+1);

    auto it = mapper.find(type);
    if (it == mapper.end()) {
        MI_IO_LOG(MI_ERROR) << "get evaluation : " << type << " mapper string failed.";
        return v;
    }

    const std::string& type_str = it->second;
    while (std::getline(in , line)) {
        if (line.empty()) {
            continue;
        }
        if (line[0] == '#') {
            continue;
        }
        std::stringstream ss(line);
        ss >> tag >> equal >> context;
        if (tag == type_str) {
            if(0 != make_version(context, v)) {
                MI_IO_LOG(MI_ERROR) << "make evaluation : " << type << " version failed.";
            }
            break;
        }
    }

    if (LUNG_NODULE == type) {
        
    }

    in.close();
    return v;
}

Version Configure::get_preprocess_version(PreprocessType type) const {
    Version v;
#ifdef WIN32
    std::ifstream in("./config/app_config");
#else
    std::ifstream in("../config/app_config");
#endif
    
    if (!in.is_open()) {
        MI_IO_LOG(MI_ERROR) << "can't open configure file when get preprocess version.";
        return v;
    }

    std::string line;
    std::string tag;
    std::string equal;
    std::string context;

    const static std::map<PreprocessType, std::string>::value_type mapper_value[] = {
        std::map<PreprocessType, std::string>::value_type(INIT_SEGMENT_MASK, "PrepInitSegMaskVersion"),
        std::map<PreprocessType, std::string>::value_type(LUNG_AI_DATA, "PrepLungAIDataVersion")
    };

    const static std::map<PreprocessType, std::string> mapper(mapper_value, mapper_value+2);

    auto it = mapper.find(type);
    if (it == mapper.end()) {
        MI_IO_LOG(MI_ERROR) << "get preprocess : " << type << " mapper string failed.";
        return v;
    }

    const std::string& type_str = it->second;
    while (std::getline(in , line)) {
        if (line.empty()) {
            continue;
        }
        if (line[0] == '#') {
            continue;
        }
        std::stringstream ss(line);
        ss >> tag >> equal >> context;
        if (tag == type_str) {
            if(0 != make_version(context, v)) {
                MI_IO_LOG(MI_ERROR) << "make preprocess : " << type << " version failed.";
            }
            break;
        }
    }

    in.close();
    return v;
}

MED_IMG_END_NAMESPACE