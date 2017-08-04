#include "PACSCommunicator.h"
#include "WorkListInfo.h"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    //OFLog::configure(OFLogger::DEBUG_LOG_LEVEL);

    PACSCommunicator test;
    if (test.initialize("config.txt")) //alternative:test.initialize("SelfSCU", 11115, "172.20.70.27", 11112, "DCM4CHEE");
    {
        test.populate_whole_work_list();
        const std::vector<WorkListInfo>& ls = test.get_work_list();
        for (auto it=ls.begin(); it != ls.end(); ++it)
        {
            std::cout << it->GetStudyInsUID() << "   " << it->GetSeriesInsUID() << std::endl;
        }

        std::string series_uid;
        while (std::cin >> series_uid)
        {
            std::string output = test.fetch_dicom(series_uid);
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