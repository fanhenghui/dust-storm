#include "io/mi_pacs_communicator.h"
#include "io/mi_io_logger.h"
#include "io/mi_configure.h"
#include "io/mi_db.h"

#include "util/mi_file_util.h"
#include "util/mi_time_util.h"
#include "util/mi_uid.h"

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

struct QueryKey {
    std::string study_uid;
    std::string series_uid;
    std::string study_date;//format: YYYYMMDD
    std::string study_time;//format: HHMMSS
    std::string patient_id;
    std::string patient_name;
    std::string modality;
    std::string accession_no;
    std::string patient_sex;
    std::string patient_birth_date;
};

inline void print_help() {
    std::cout << "COMMAND DESCRIPTION:\n"
    << "\nquery [q]\n"
    << "\t-level [l]\n"
    << "\t\tquery level. format: series | study | patient[p].\n"
    << "\t-study_uid\n"
    << "\t\tstudy uid.\n"
    << "\t-series_uid\n"
    << "\t\tseries uid.\n"
    << "\t-study_date\n"
    << "\t\tstudy date. format: YYYYMMDD | YYYYMMDD-YYYYMMDD.\n"
    << "\t-patient_name\n"
    << "\t\tpatient name.\n"
    << "\t-patient_id\n"
    << "\t\tpatient id.\n"
    << "\t-patient_sex\n"
    << "\t\tpatient sex. format: M | F\n"
    << "\t-patient_birth_date\n"
    << "\t\tpatient birth date. format: YYYYMMDD | YYYYMMDD-YYYYMMDD.\n"
    << "\t-modality\n"
    << "\t\tmodality. format: CT | MR | CR | PT | RT_STRUCT.\n"
    << "\t-accession_no\n"
    << "\t\taccession number.\n"
    << "\nretrieve [r]\n"
    << "\t-series_uid [s]\n"
    << "\t\t series uid.\n"
    << "\nexit\n"
    << "\texit console.\n"
    << "\nhelp\n"
    << "\tprint help message.\n";
} 

int pacs_ut(int argc, char* argv[]) {
#ifndef WIN32
    if(0 != chdir(dirname(argv[0]))) {

    }
#endif
    MI_IO_LOG(MI_INFO) << "Hello PACS UT.";

    std::string PACSServerAETitle = "DCM4CHEE";
    std::string PACSServerHost = "192.168.199.107";
    unsigned short PACSServerPort = 11112;
    std::string PACSClientAETitle = "DBS";
    unsigned short PACSClientPort = 11115;

    Configure::instance()->get_pacs_info(PACSServerAETitle, PACSServerHost, PACSServerPort, PACSClientAETitle, PACSClientPort);

    PACSCommunicator pacs_comm;
    if(-1 == pacs_comm.connect(PACSServerAETitle, PACSServerHost, PACSServerPort, PACSClientAETitle, PACSClientPort)) {
        return -1;
    }

    DB db;
    std::string ip_port,user,pwd,db_name;
    Configure::instance()->get_db_info(ip_port, user, pwd, db_name);
    if(0 != db.connect(user, ip_port, pwd, db_name) ) {
        MI_IO_LOG(MI_FATAL) << "connect to db failed.";
        return -1;
    }

    std::cout << "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n";
    std::cout << "                  PACS communication Test\n";
    std::cout << "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n";
    print_help();

    std::string cmd;
    while (std::getline(std::cin, cmd)) {
        std::vector<std::string> item;
        boost::split(item, cmd, boost::is_any_of(" "));

        bool invalid_cmd = false;

#define CHECK_INDEX  if(i>=item.size()-1) { \
        MI_IO_LOG(MI_WARNING) << "invalid cmd"; \
        break;} \

#define INVALID_CMD invalid_cmd = true; break;

        if (item[0] == "query" || item[0] == "q") {
            QueryKey qkey;
            int qlevel = -1;//0 study, 1 series, 2 patient
            for (size_t i=1; i<item.size(); ++i) {
                if (item[i] == "-level" || item[i] == "-l") {
                    CHECK_INDEX;
                    if (item[i+1] == "study") {
                        qlevel = 0;
                        ++i;
                    } else if(item[i+1] == "series") {
                        qlevel = 1;
                        ++i;
                    } else if(item[i+1] == "patient" || item[i+1] == "p") {
                        qlevel = 2;
                        ++i;
                    } else {
                        INVALID_CMD;
                    }
                } else if (item[i] == "-study_uid") {
                    CHECK_INDEX;
                    qkey.study_uid = item[++i];
                } else if (item[i] == "-series_uid") {
                    CHECK_INDEX;
                    qkey.series_uid = item[++i];
                } else if (item[i] == "-study_date") {
                    CHECK_INDEX;
                    qkey.study_date = item[++i];       
                } else if (item[i] == "-patient_id") {
                    CHECK_INDEX;
                    qkey.patient_id = item[++i];       
                } else if (item[i] == "-patient_name") {
                    CHECK_INDEX;
                    qkey.patient_name = item[++i];       
                } else if (item[i] == "-patient_birth_date") {
                    CHECK_INDEX;
                    qkey.patient_birth_date = item[++i];       
                } else if (item[i] == "-modality") {
                    CHECK_INDEX;
                    qkey.modality = item[++i];       
                } else if (item[i] == "-accession_no") {
                    CHECK_INDEX;
                    qkey.accession_no = item[++i];       
                } else if (item[i] == "-patient_sex") {
                    CHECK_INDEX;
                    qkey.patient_sex = item[++i];       
                }
            }

            if (qlevel != 0 && qlevel != 1 && qlevel != 2) {
                invalid_cmd = true;
            }

            if (!invalid_cmd) {
                StudyInfo study_key;
                SeriesInfo series_key;
                PatientInfo patient_key;

                study_key.study_uid = qkey.study_uid;
                study_key.study_date = qkey.study_date;
                study_key.study_time = qkey.study_time;
                study_key.accession_no = qkey.accession_no;

                series_key.series_uid = qkey.series_uid;
                series_key.modality = qkey.modality;

                patient_key.patient_name = qkey.patient_name;
                patient_key.patient_id = qkey.patient_id;
                patient_key.patient_birth_date = qkey.patient_birth_date;
                patient_key.patient_sex = qkey.patient_sex;


                if (qlevel == 0) {//study
                    MI_IO_LOG(MI_WARNING) << "query level: study, query key: " << "\n"
                    << "study_uid: " << qkey.study_uid << "\n" 
                    << "study_date: " << qkey.study_date << "\n"
                    << "study_time: " << qkey.study_time << "\n"
                    << "accession_no: " << qkey.accession_no << "\n"
                    << "patient_name: " << qkey.patient_name << "\n" 
                    << "patient_id: " << qkey.patient_id << "\n"
                    << "patient_birth_date: " << qkey.patient_birth_date << "\n"
                    << "patient_sex: " << qkey.patient_sex << "\n";

                    std::vector<StudyInfo> study_infos;
                    std::vector<PatientInfo> patient_infos;  
                    if(-1 != pacs_comm.query_study(patient_key, study_key, &patient_infos, &study_infos)) {
                        MI_IO_LOG(MI_WARNING) << "<><><><><><> PCAS QUERY RESULT <><><><><><>\n";

                        for (size_t i = 0; i < study_infos.size(); ++i) {
                            const StudyInfo& study_info = study_infos[i];
                            const PatientInfo& patient_info = patient_infos[i];
        
                            MI_IO_LOG(MI_DEBUG) << i <<": " << std::endl
                            << "study_uid: " << study_info.study_uid << std::endl
                            << "study_id: " << study_info.study_id << std::endl
                            << "study_date: " << study_info.study_date << std::endl
                            << "study_time: " << study_info.study_time << std::endl
                            << "accession_no: " << study_info.accession_no << std::endl
                            << "study_desc: " << study_info.study_desc << std::endl
                            << "num_instance: " << study_info.num_instance << std::endl
                            << "num_series: " << study_info.num_series << std::endl
                            << "patient_id: " << patient_info.patient_id << std::endl
                            << "patient_name: " << patient_info.patient_name << std::endl
                            << "patient_sex: " << patient_info.patient_sex << std::endl
                            << "patient_birth_date: " << patient_info.patient_birth_date << std::endl
                            << std::endl;
                        }
                    } else {
                        MI_IO_LOG(MI_ERROR) << "PACS commucation query study failed.";
                    }
                } else if (qlevel == 1) {//series
                    MI_IO_LOG(MI_WARNING) << "query level: series, query key: " << "\n"
                    << "series_uid: " << qkey.series_uid << "\n" 
                    << "modality: " << qkey.modality << "\n"
                    << "study_uid: " << qkey.study_uid << "\n" 
                    << "study_date: " << qkey.study_date << "\n"
                    << "study_time: " << qkey.study_time << "\n"
                    << "accession_no: " << qkey.accession_no << "\n"
                    << "patient_name: " << qkey.patient_name << "\n" 
                    << "patient_id: " << qkey.patient_id << "\n"
                    << "patient_birth_date: " << qkey.patient_birth_date << "\n"
                    << "patient_sex: " << qkey.patient_sex << "\n";

                    std::vector<StudyInfo> study_infos;
                    std::vector<SeriesInfo> series_infos;
                    std::vector<PatientInfo> patient_infos;
                    if(-1 != pacs_comm.query_series(patient_key, study_key, series_key, &patient_infos, &study_infos, &series_infos)) {
                        MI_IO_LOG(MI_WARNING) << "<><><><><><> PCAS QUERY RESULT <><><><><><>\n";

                        for (size_t i = 0; i < study_infos.size(); ++i) {
                            const StudyInfo& study_info = study_infos[i];
                            const SeriesInfo& series_info = series_infos[i];
                            const PatientInfo& patient_info = patient_infos[i];
        
                            MI_IO_LOG(MI_DEBUG) << i <<": " << std::endl
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
                        }
                    } else {
                        MI_IO_LOG(MI_ERROR) << "PACS commucation query study failed.";
                    }
                } else if (qlevel == 2) {//patient
                    MI_IO_LOG(MI_WARNING) << "query level: patient, query key: " << "\n"
                    << "patient_name: " << qkey.patient_name << "\n" 
                    << "patient_id: " << qkey.patient_id << "\n"
                    << "patient_sex: " << qkey.patient_sex << "\n"
                    << "patient_birth_date: " << qkey.patient_birth_date << "\n";

                    std::vector<PatientInfo> patient_infos;  
                    if(-1 != pacs_comm.query_patient(patient_key, &patient_infos)) {
                        MI_IO_LOG(MI_WARNING) << "<><><><><><> PCAS QUERY RESULT <><><><><><>\n";

                        for (size_t i = 0; i < patient_infos.size(); ++i) {
                            const PatientInfo& patient_info = patient_infos[i];
        
                            MI_IO_LOG(MI_DEBUG) << i <<": " << std::endl
                            << "patient_id: " << patient_info.patient_id << std::endl
                            << "patient_name: " << patient_info.patient_name << std::endl
                            << "patient_sex: " << patient_info.patient_sex << std::endl
                            << "patient_birth_date: " << patient_info.patient_birth_date << std::endl
                            << std::endl;
                        }
                    } else {
                        MI_IO_LOG(MI_ERROR) << "PACS commucation query study failed.";
                    }
                }
            }
        } else if (item[0] == "retrieve" || item[0] == "r") {
            std::string series_uid = "";
            for (size_t i=1; i<item.size(); ++i) {
                if (item[i] == "-series_uid" || item[i] == "-s") {
                    CHECK_INDEX;
                    series_uid = item[++i]; 
                }
            }

            if (!invalid_cmd) {
                std::string path = Configure::instance()->get_db_path();
                FileUtil::make_direction(path);
                path += "/instance";
                FileUtil::make_direction(path);
                path += "/" + TimeUtil::current_date();
                FileUtil::make_direction(path);
                path += "/" + UUIDGenerator::uuid();
                FileUtil::make_direction(path);
                MI_IO_LOG(MI_INFO) << "try retrieve series: " << series_uid << " to path: " << path;
                std::vector<InstanceInfo> instance_infos;
                std::vector<PatientInfo> patient_infos;
                std::vector<StudyInfo> study_infos;
                std::vector<SeriesInfo> series_infos;
                PatientInfo pkey;
                StudyInfo studykey;
                SeriesInfo serieskey;
                serieskey.series_uid = series_uid;
                if (-1 != pacs_comm.query_series(pkey, studykey, serieskey, &patient_infos, &study_infos, &series_infos));
                if( -1 != pacs_comm.retrieve_series(series_uid, path, &instance_infos) ) {
                    std::vector<UserInfo> user_infos;
                    if (!user_infos.empty()) {
                        db.insert_series(patient_infos[0], study_infos[0], series_infos[0], instance_infos);
                    }
                    
                } else {
                    MI_IO_LOG(MI_ERROR) << "PACS commucation retrieve series failed.";
                }
            }
        } else if (item[0] == "exit") {
            pacs_comm.disconnect();
            MI_IO_LOG(MI_INFO) << "Byby.";            
            break;
        } else if (item[0] == "help"){
            print_help();
        } else {
            invalid_cmd = true;
        }

        if (invalid_cmd) {
            MI_IO_LOG(MI_WARNING) << "invalid cmd";            
        }
    }   

    return 0;
}