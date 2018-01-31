#include "io/mi_pacs_communicator.h"
#include "io/mi_io_logger.h"
#include "io/mi_configure.h"

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

int pacs_ut(int argc, char* argv[]) {
#ifndef WIN32
    if(0 != chdir(dirname(argv[0]))) {

    }
#endif

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
    std::vector<DcmInstanceInfo> instance_infos;
    std::vector<DcmInfo> dcm_infos;
    // if(-1 == pacs_comm.retrieve_all_series(dcm_infos)) {
    //     return -1;
    // }

    QueryKey query_key;
    //query_key.study_date = "19990201-20151010";
    //query_key.patient_id = "A*";
    //query_key.accession_number = "2819497684894126";
    //query_key.patient_name = "A*";
    //query_key.modality = "CT";
    //query_key.patient_sex = "M";

    if(-1 == pacs_comm.query_study(dcm_infos, query_key)) {
        return -1;
    }

    MI_IO_LOG(MI_WARNING) << "<><><><><><> STUDY QUERY RESULT <><><><><><>\n";
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
        << "accession_number: " << info.accession_number << std::endl
        << "series_no: " << info.series_no << std::endl
        << "institution: " << info.institution << std::endl
        << "series_desc: " << info.series_desc << std::endl
        << "study_desc: " << info.study_desc << std::endl
        << "number_of_instance: " << info.number_of_instance << std::endl
        << "number_of_series: " << info.number_of_series << std::endl
        << std::endl;
    }

    if(-1 == pacs_comm.query_series(dcm_infos, query_key)) {
        return -1;
    }

    MI_IO_LOG(MI_WARNING) << "<><><><><><> SERIES QUERY RESULT <><><><><><>\n";
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
        << "accession_number: " << info.accession_number << std::endl
        << "series_no: " << info.series_no << std::endl
        << "institution: " << info.institution << std::endl
        << "series_desc: " << info.series_desc << std::endl
        << "study_desc: " << info.study_desc << std::endl
        << "number_of_instance: " << info.number_of_instance << std::endl
        << "number_of_series: " << info.number_of_series << std::endl
        << std::endl;
    }

    int query_id = -1;
    while (std::cin >> query_id) {
        if (query_id == -1) {
            MI_IO_LOG(MI_DEBUG) << "<><><><><><> QUERY RESULT <><><><><><>";
            int id = 0;
            for (auto it = dcm_infos.begin(); it != dcm_infos.end(); ++it) {
                const std::string series_id = (*it).series_id;
                MI_IO_LOG(MI_DEBUG) << id++ << series_id;
            }
            MI_IO_LOG(MI_DEBUG) << "<><><><><><> QUERY RESULT <><><><><><>";
        } else if(query_id == -2) {
            MI_IO_LOG(MI_INFO) << "query all series again.";
            if(-1 == pacs_comm.query_series(dcm_infos, QueryKey())) {
                return -1;
            }
            MI_IO_LOG(MI_DEBUG) << "<><><><><><> QUERY RESULT <><><><><><>";
            int id = 0;
            for (auto it = dcm_infos.begin(); it != dcm_infos.end(); ++it) {
                const std::string series_id = (*it).series_id;
                MI_IO_LOG(MI_DEBUG) << "ID: "<< id++ << "  SeriesID: "<< series_id;
            }
            MI_IO_LOG(MI_DEBUG) << "<><><><><><> QUERY RESULT <><><><><><>";
        } else if(query_id == -3) {
            pacs_comm.disconnect();
            break;
        } else if(query_id >= 0 && query_id < id) {
            MI_IO_LOG(MI_DEBUG) << "retrieve series: " << dcm_infos[query_id].series_id;
            instance_infos.clear();
            pacs_comm.retrieve_series(dcm_infos[query_id].series_id, "/home/wangrui22/data/cache", &instance_infos);
        } else {
            MI_IO_LOG(MI_WARNING) << "invalid query ID.";
        }
    }

    MI_IO_LOG(MI_INFO) << "DONE.";

    return 0;
}