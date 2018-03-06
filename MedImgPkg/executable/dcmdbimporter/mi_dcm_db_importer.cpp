#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <termios.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "boost/filesystem.hpp"

#ifndef WIN32
#include <dlfcn.h>
#endif

#include "log/mi_logger.h"
#include "util/mi_file_util.h"
#include "io/mi_db.h"
#include "io/mi_configure.h"

#include "mi_dcm_file_browser.h"
#include "util/mi_time_util.h"
#include "util/mi_uid.h"

using namespace medical_imaging;

int main(int argc, char* argv[]) {

    const std::string log_config_file = "../config/log_cofig";
    Logger::instance()->bind_config_file(log_config_file);
    Logger::instance()->set_file_name_format("logs/dcm-db-importer-log-%Y-%m-%d_%H-%M-%S.%N.log");
    Logger::instance()->set_file_direction("");
    Logger::instance()->initialize();
    
    DB db;
    std::string ip_port,user,pwd,db_name;
    Configure::instance()->get_db_info(ip_port, user, pwd, db_name);
    if(0 != db.connect(user, ip_port, pwd, db_name) ) {
        MI_LOG(MI_ERROR) << "connect to db failed.";
        return 0;
    }

    int ch;
    std::string src_direction;
    while ((ch = getopt(argc, argv, "s:")) != -1) {
        switch (ch) {
            case 's':
                src_direction = std::string(optarg);
                break;
            default:
                break;
        }
    }

    if (src_direction.empty()){
        MI_LOG(MI_ERROR) << "DICOM source direction empty.";
        return 0;
    }

    //------------------------------//
    // file browser
    //------------------------------//
    DcmFileBrowser browser;
    browser.browse(src_direction);

    //------------------------------//
    // copy & insert into DB
    //------------------------------//
    size_t count = 0;
    size_t sum = browser._series_infos.size();
    const std::string db_path = Configure::instance()->get_db_path();
    const std::string dcm_path = db_path + "/instance/" + TimeUtil::current_date();

    FileUtil::make_direction(db_path);
    FileUtil::make_direction(db_path + "/instance/");
    FileUtil::make_direction(dcm_path);
    
    for (auto it = browser._series_infos.begin(); it!= browser._series_infos.end(); ++it) {
        const std::string& series_uid = it->first;
        StudyInfo& study_info = browser._study_infos[series_uid];
        SeriesInfo& series_info = browser._series_infos[series_uid];
        PatientInfo& patient_info = browser._patient_infos[series_uid];

        //print debug
        MI_LOG(MI_INFO) << ++count <<"/" << sum << " : " << std::endl
                            << "study_uid: " << study_info.study_uid << std::endl
                            << "study_id: " << study_info.study_id << std::endl
                            << "study_date: " << study_info.study_date << std::endl
                            << "study_time: " << study_info.study_time << std::endl
                            << "accession_no: " << study_info.accession_no << std::endl
                            << "study_desc: " << study_info.study_desc << std::endl
                            << "num_instance(study): " << study_info.num_instance << std::endl
                            << "num_series: " << study_info.num_series << std::endl
                            << "series_uid: " << series_info.series_uid << std::endl
                            << "series_no: " << series_info.series_no << std::endl
                            << "modality: " << series_info.modality << std::endl
                            << "series_desc: " << series_info.series_desc << std::endl
                            << "institution: " << series_info.institution << std::endl
                            << "num_instance(series): " << series_info.num_instance << std::endl
                            << "patient_id: " << patient_info.patient_id << std::endl
                            << "patient_name: " << patient_info.patient_name << std::endl
                            << "patient_sex: " << patient_info.patient_sex << std::endl
                            << "patient_birth_date: " << patient_info.patient_birth_date << std::endl
                            << std::endl;


        const std::string series_map_path = dcm_path + "/" + UUIDGenerator::uuid();
        FileUtil::make_direction(series_map_path);

        std::vector<InstanceInfo>& instances = browser._instance_infos[series_uid];
        for (size_t i = 0; i < instances.size(); ++i) {
            const std::string dst = series_map_path + "/" + series_info.modality + "." 
                + instances[i].sop_instance_uid + ".dcm";
            if (0 != FileUtil::copy_file(instances[i].file_path, dst)) {
                MI_LOG(MI_ERROR) << "copy instance :" << instances[i].file_path << 
                "to " << dst << " failed.";
                continue;
            }
            instances[i].file_path = dst;
        }
        series_info.num_instance = (int)instances.size();
        //insert to DB
        if(0 != db.insert_series(patient_info, study_info, series_info, instances)) {
            MI_LOG(MI_ERROR) << "insert series: " << series_uid << " into db failed.";
            return 0;
        }
    }

    MI_LOG(MI_INFO) << "DICOM import to DB success.";
    
    return 0;
}