#ifndef MED_IMG_REVIEW_SERVER_H
#define MED_IMG_REVIEW_SERVER_H

#include "mi_review_common.h"
#include <string>
#include "boost/thread/mutex.hpp"

MED_IMG_BEGIN_NAMESPACE

class ReviewConfig {
public:
    static ReviewConfig* instance();
    ~ReviewConfig();

    std::string get_config_root() const;
    std::string get_log_config_file() const;
    std::string get_test_data_root() const;
    std::string get_db_pwd() const;
    int get_expected_fps() const;
    float get_nodule_possibility_threshold() const;

private:
    ReviewConfig();
    static ReviewConfig* _instance;
    static boost::mutex _mutex;

private:
    std::string _test_data_root;
    std::string _db_wpd;
    int _expected_fps;
    float _nodule_p_th;
};


MED_IMG_END_NAMESPACE


#endif