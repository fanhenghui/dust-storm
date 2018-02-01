#include "io/mi_pacs_communicator.h"
#include "io/mi_io_logger.h"
#include "io/mi_configure.h"

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

inline void print_help() {
    std::cout << "COMMAND DESCRIPTION:\n"
    << "\nquery\n"
    << "\t-level [l]\n"
    << "\t\tquery level. format: series | study | patient.\n"
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
    << "\nretrieve\n"
    << "\t-series-uid\n"
    << "\t\t series uid.\n"
    << "\t-path\n"
    << "\t\t file path.\n"
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

        if (item[0] == "query") {
            QueryKey qkey;
            int qlevel = -1;//0 study, 1 series
            for (size_t i=1; i<item.size(); ++i) {
                if (item[i] == "-level" || item[i] == "-l") {
                    CHECK_INDEX;
                    if (item[i+1] == "study") {
                        qlevel = QueryLevel::STUDY;
                        ++i;
                    } else if(item[i+1] == "series") {
                        qlevel = QueryLevel::SERIES;
                        ++i;
                    } else if(item[i+1] == "patient") {
                        qlevel = QueryLevel::PATIENT;
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

                MI_IO_LOG(MI_WARNING) << "query key: [\n"
                <<  "\t study_uid: " << qkey.study_uid << "\n"
                <<  "\t series_uid: " << qkey.series_uid << "\n"
                <<  "\t study_date: " << qkey.study_date << "\n"
                <<  "\t study_time: " << qkey.study_time << "\n"
                <<  "\t patient_id: " << qkey.patient_id << "\n"
                <<  "\t patient_name: " << qkey.patient_name << "\n"
                <<  "\t modality: " << qkey.modality << "\n"
                <<  "\t accession_no: " << qkey.accession_no << "\n"
                <<  "\t patient_sex: " << qkey.patient_sex << "\n"
                <<  "\t patient_birth_date: " << qkey.patient_birth_date << "\n"
                << "]\n";

                std::vector<DcmInfo> dcm_infos;    
                if(-1 != pacs_comm.query(dcm_infos, qkey, (QueryLevel)qlevel) ) {
                    MI_IO_LOG(MI_WARNING) << "<><><><><><> PCAS QUERY RESULT <><><><><><>\n";

                    int id = 0;
                    for (auto it = dcm_infos.begin(); it != dcm_infos.end(); ++it) {
                        const std::string series_id = (*it).series_id;
                        const DcmInfo& info = *it;
        
                        MI_IO_LOG(MI_DEBUG) << id++ <<": " << std::endl
                        << "study_id: " << info.study_id << std::endl
                        << "series_id: " << info.series_id << std::endl
                        << "study_date: " << info.study_date << std::endl
                        << "study_time: " << info.study_time << std::endl
                        << "patient_id: " << info.patient_id << std::endl
                        << "patient_name: " << info.patient_name << std::endl
                        << "patient_sex: " << info.patient_sex << std::endl
                        << "patient_birth_date: " << info.patient_birth_date << std::endl
                        << "modality: " << info.modality << std::endl
                        << "accession_no: " << info.accession_no << std::endl
                        << "series_no: " << info.series_no << std::endl
                        << "institution: " << info.institution << std::endl
                        << "series_desc: " << info.series_desc << std::endl
                        << "study_desc: " << info.study_desc << std::endl
                        << "number_of_instance: " << info.number_of_instance << std::endl
                        << "number_of_series: " << info.number_of_series << std::endl
                        << std::endl;
                    }
                } else {
                    MI_IO_LOG(MI_ERROR) << "PACS commucation query failed.";
                }
            }

        } else if (item[0] == "retrieve") {
            std::string series_uid = "";
            std::string path = "";
            for (size_t i=1; i<item.size(); ++i) {
                if (item[i] == "-series_uid") {
                    CHECK_INDEX;
                    series_uid = item[++i]; 
                } else if(item[i] == "-path") {
                    CHECK_INDEX;
                    path = item[++i]; 
                }
            }

            if (!invalid_cmd) {
                MI_IO_LOG(MI_INFO) << "try retrieve series: " << series_uid << " to path: " << path;
                std::vector<DcmInstanceInfo> instance_infos;
                if( -1 != pacs_comm.retrieve_series(series_uid, path, &instance_infos) ) {
                    
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