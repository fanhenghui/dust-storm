#ifndef MED_IMG_APPCOMMON_MI_APP_CONFIG_H
#define MED_IMG_APPCOMMON_MI_APP_CONFIG_H

#include "appcommon/mi_app_common_export.h"
#include <string>
#include "boost/thread/mutex.hpp"

MED_IMG_BEGIN_NAMESPACE

class AppConfig {
public:
    static AppConfig* instance();
    ~AppConfig();

    std::string get_config_root() const;
    std::string get_log_config_file() const;
    std::string get_test_data_root() const;

    void get_cache_db_info(std::string& ip_port, std::string& user, std::string& pwd, std::string& db_name, std::string& path);
    void get_db_info(std::string& ip_port, std::string& user, std::string& pwd, std::string& db_name);
    void get_db_server_host(std::string& ip, std::string& port);

    int get_expected_fps() const;
    float get_nodule_possibility_threshold() const;
private:
    AppConfig();
    static AppConfig* _instance;
    static boost::mutex _mutex;

private:
    std::string _test_data_root;
    float _nodule_possibility_threshold;

    //DB info
    std::string _db_ip;
    std::string _db_port;
    std::string _db_user;
    std::string _db_pwd;
    std::string _db_name;

    //DB server (ip is the same with DB)
    std::string _db_server_port;

    //cache DB info
    std::string _cache_db_ip;
    std::string _cache_db_port;
    std::string _cache_db_user;
    std::string _cache_db_pwd;
    std::string _cache_db_name;
    std::string _cache_db_path;

    int _expected_fps;
};

MED_IMG_END_NAMESPACE


#endif