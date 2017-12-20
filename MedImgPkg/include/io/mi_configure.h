#ifndef MEDIMG_IO_MI_CONFIGURE_H
#define MEDIMG_IO_MI_CONFIGURE_H

#include <string>
#include "io/mi_io_export.h"
#include "boost/thread/mutex.hpp"

MED_IMG_BEGIN_NAMESPACE

enum ProcessingUnitType {
    CPU = 0,
    GPU,
};

class IO_Export Configure {
public:
    static Configure* instance();
    ~Configure();

    //For warm start/update
    void refresh();

    std::string get_config_root() const;
    std::string get_test_data_root() const;

    //For LOG module
    std::string get_log_config_file() const;

    //For DB module
    void get_cache_db_info(std::string& ip_port, std::string& user, std::string& pwd, std::string& db_name, std::string& path) const;
    void get_db_info(std::string& ip_port, std::string& user, std::string& pwd, std::string& db_name) const;
    std::string get_db_path() const;
    void get_db_server_host(std::string& ip, std::string& port) const;

    //For Render Module
    int get_expected_fps() const;

    //For AI Module
    float get_evaluation_probability_threshold() const;
    int get_evaluation_limit() const;
    std::string get_pytorch_path() const;
    std::string get_py_interface_path() const;

    //For DBS <=> PACS
    void get_pacs_info(std::string& server_ae_title, std::string& server_host ,unsigned short& server_port,
    std::string& client_ae_title, unsigned short& client_port);

    //GPU/CPU MPR rendering(Just for Windows client)
    ProcessingUnitType get_processing_unit_type() const;
    void set_processing_unit_type(ProcessingUnitType type);//For testing
private:
    Configure();
    void init();

    static Configure* _instance;
    static boost::mutex _mutex;

private:
    //For debug
    std::string _test_data_root;

    //DB info
    std::string _db_ip;
    std::string _db_port;
    std::string _db_user;
    std::string _db_pwd;
    std::string _db_name;
    std::string _db_path;

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

    //AI server
    float _nodule_possibility_threshold;
    int _evaluation_limit;
    std::string _pytorch_path;//anaconda/envs
    std::string _py_interface_path;

    //DBS <=> PACS
    std::string _pacs_server_ae_title;
    std::string _pacs_server_host;
    unsigned short _pacs_server_port;
    std::string _pacs_client_ae_title;
    unsigned short _pacs_client_port;  

    //GPU/CPU MPR rendering
    ProcessingUnitType _processing_unit_type;

    //
};

MED_IMG_END_NAMESPACE
#endif