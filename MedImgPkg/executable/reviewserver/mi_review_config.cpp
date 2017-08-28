#include "mi_review_config.h"
#include <fstream>
#include <sstream>

MED_IMG_BEGIN_NAMESPACE

ReviewConfig* ReviewConfig::_instance = nullptr;
boost::mutex ReviewConfig::_mutex;

ReviewConfig::ReviewConfig()
{
    std::ifstream in("../config/configure.txt");
    if(!in.is_open()){
        //TODO ERROR LOG
        return;
    }

    std::string line;
    std::string tag;
    std::string equal;
    std::string context;
    while(std::getline(in , line)){
        std::stringstream ss(line);
        ss >> tag >> equal >> context;
        if(tag == "TestDataRoot"){
            _test_data_root = context;
            break;
        }
    }

    //TODO CHECK path valid
}

ReviewConfig::~ReviewConfig()
{
    
}

ReviewConfig* ReviewConfig::instance()
{
    if(nullptr == _instance){
        boost::mutex::scoped_lock locker(_mutex);
        if(nullptr == _instance){
            _instance = new ReviewConfig();
        }
    }

    return _instance;
}

std::string ReviewConfig::get_test_data_root() const
{
    return _test_data_root;
}


MED_IMG_END_NAMESPACE