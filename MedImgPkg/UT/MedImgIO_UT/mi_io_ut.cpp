#include "io/mi_pacs_communicator.h"
#include "io/mi_worklist_info.h"
#include "dcmtk/oflog/oflog.h"
#include <iostream>
#include <string>
#include <vector>

#ifdef WIN32
#else
#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#endif

using namespace medical_imaging;

int main(int argc, char* argv[])
{
    #ifndef WIN32
    chdir(dirname(argv[0]));
    #endif

    OFLog::configure(OFLogger::DEBUG_LOG_LEVEL);

    PACSCommunicator test;
    if (test.initialize("../Config/pacs_config.txt")) //alternative:test.initialize("SelfSCU", 11115, "172.20.70.27", 11112, "DCM4CHEE");
    {
        test.populate_whole_work_list();
        const std::vector<WorkListInfo>& ls = test.get_work_list();
        for (auto it=ls.begin(); it != ls.end(); ++it)
        {
            std::cout << it->GetStudyInsUID() << "   " << it->GetSeriesInsUID() << std::endl;
        }

        int series_uid_idx = 1;
        // std::string output = test.fetch_dicom(ls[series_uid_idx]);
        // if (output == "")
        // {
        //     //break;
        // }
        // else
        // {
        //     std::cout << "Output directory : " << output << std::endl;
        // }
        while (std::cin >> series_uid_idx)
        {
            std::string output = test.fetch_dicom(ls[series_uid_idx]);
            if (output == "")
            {
                break;
            }
            else
            {
                std::cout << "Output directory : " << output << std::endl;
            }
        }
    }
    return 0;
}