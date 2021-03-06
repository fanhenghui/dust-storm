#include "io/mi_db.h"
#include "io/mi_io_logger.h"
#include "io/mi_configure.h"
#include "io/mi_io_logger.h"

#include <boost/algorithm/string.hpp>
#include <iostream>
#include <string>
#include <vector>

#ifdef WIN32
#else
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#endif

using namespace medical_imaging;


int db_ut(int argc, char* argv[]) {
#ifndef WIN32
    if(0 != chdir(dirname(argv[0]))) {

    }
#endif
    MI_IO_LOG(MI_INFO) << "Hello DB Test.";

    DB db;
    std::string ip_port,user,pwd,db_name;
    Configure::instance()->get_db_info(ip_port, user, pwd, db_name);
    if(0 != db.connect(user, ip_port, pwd, db_name) ) {
        MI_IO_LOG(MI_FATAL) << "connect to db failed.";
        return -1;
    }

    // PatientInfo patient_info;
    // patient_info.patient_name = "OJBK123";
    // patient_info.patient_id = "";
    // patient_info.patient_birth_date = "";
    // patient_info.patient_sex = "";
    // db.insert_patient(patient_info);

    // MI_IO_LOG(MI_DEBUG) << "patient id: " << patient_info.id;

    // PatientInfo key;
    // key.md5 = patient_info.md5;
    // std::vector<PatientInfo> res;
    // db.query_patient(key, &res);

    // if(!res.empty()) {
    //     MI_IO_LOG(MI_INFO) << res.size();
    //     MI_IO_LOG(MI_INFO) << res[0].patient_name;
    //     MI_IO_LOG(MI_INFO) << res[0].patient_id;
    //     MI_IO_LOG(MI_INFO) << res[0].patient_birth_date;
    //     MI_IO_LOG(MI_INFO) << res[0].patient_sex;
    //     MI_IO_LOG(MI_INFO) << res[0].md5;
    // }
    
    // MI_IO_LOG(MI_INFO) << "BYBY.";

    std::string date = "19890103";
    MI_IO_LOG(MI_INFO) << atoi(date.substr(4, 2).c_str());
    
    std::vector<UserInfo> user_infos;
    UserInfo ukey;
    db.query_user(ukey, &user_infos);
    if (!user_infos.empty()) {
        for (auto it = user_infos.begin(); it != user_infos.end(); ++it) {
            UserInfo& info = *it;
            MI_IO_LOG(MI_INFO) << "user: " << info.id << ", " << info.name << ", " << info.role_fk; 
        }
    } else {
        MI_IO_LOG(MI_WARNING) << "query user null.";
    }
    return 0;
}