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
    if(-1 == pacs_comm.query_series(dcm_infos, query_key)) {
        return -1;
    }

    MI_IO_LOG(MI_DEBUG) << "<><><><><><> QUERY RESULT <><><><><><>";
    int id = 0;
    for (auto it = dcm_infos.begin(); it != dcm_infos.end(); ++it) {
        const std::string series_id = (*it).series_id;
        MI_IO_LOG(MI_DEBUG) << id++ <<": series_id" << (*it).series_id << "\n"
            << "study_id: " << (*it).study_id << "\n"
            << "instance_number: " << (*it).number_of_instance << "\n"
            << ", patient_id: " << (*it).patient_id << "\n"
            << ", patient_name: " << (*it).patient_name << "\n"
            << ", accession_number: " << (*it).accession_number << "\n\n"; 
    }

    MI_IO_LOG(MI_DEBUG) << "<><><><><><> QUERY RESULT <><><><><><>";

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