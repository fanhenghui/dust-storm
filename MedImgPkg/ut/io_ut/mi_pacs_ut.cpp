#include "io/mi_pacs_communicator.h"
#include "io/mi_io_logger.h"
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

    const std::string PACSServerAETitle = "DCM4CHEE";
    const std::string PACSServerHost = "192.168.199.107";
    const unsigned short PACSServerPort = 11112;
    const std::string PACSClientAETitle = "DBS";
    const unsigned short PACSClientPort = 11115;

    PACSCommunicator pacs_comm;
    if(-1 == pacs_comm.connect(PACSServerAETitle, PACSServerHost, PACSServerPort, PACSClientAETitle, PACSClientPort)) {
        return -1;
    }

    std::vector<DcmInfo> dcm_infos;
    // if(-1 == pacs_comm.retrieve_all_series(dcm_infos)) {
    //     return -1;
    // }

    if(-1 == pacs_comm.retrieve_series(dcm_infos, "19990201", "20050101")) {
        return -1;
    }

    MI_IO_LOG(MI_DEBUG) << "<><><><><><> QUERY RESULT <><><><><><>";
    int id = 0;
    for (auto it = dcm_infos.begin(); it != dcm_infos.end(); ++it) {
        const std::string series_id = (*it).series_id;
        MI_IO_LOG(MI_DEBUG) << id++ << series_id;
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
            if(-1 == pacs_comm.retrieve_all_series(dcm_infos)) {
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
            MI_IO_LOG(MI_DEBUG) << "fetch series: " << dcm_infos[query_id].series_id;
            pacs_comm.fetch_series(dcm_infos[query_id].series_id, "/home/wangrui22/data/cache");
        } else {
            MI_IO_LOG(MI_WARNING) << "invalid query ID.";
        }
    }

    MI_IO_LOG(MI_INFO) << "DONE.";

    return 0;
}