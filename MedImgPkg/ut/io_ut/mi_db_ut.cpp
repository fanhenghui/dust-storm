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

    PatientInfo patient_info;
    patient_info.patient_name = "OJBK1";
    patient_info.patient_id = "3344";
    patient_info.patient_birth_date = "20000101";
    patient_info.patient_sex = "M";
    db.insert_patient(patient_info);

    MI_IO_LOG(MI_DEBUG) << "patient id: " << patient_info.id;
    
    MI_IO_LOG(MI_INFO) << "BYBY.";
    return 0;
}