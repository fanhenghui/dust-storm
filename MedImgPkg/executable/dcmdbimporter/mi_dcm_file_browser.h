#ifndef MI_DCM_FILE_BROWSER_H
#define MI_DCM_FILE_BROWSER_H

#include <string>
#include "io/mi_dicom_info.h"

namespace medical_imaging{
class DcmFileBrowser {
public:
    DcmFileBrowser() {}

    ~DcmFileBrowser() {}

    int browse(const std::string& direction);

public:
    std::map<std::string, std::vector<InstanceInfo>> _instance_infos;
    std::map<std::string, StudyInfo> _study_infos;
    std::map<std::string, SeriesInfo> _series_infos;
    std::map<std::string, PatientInfo> _patient_infos;
};
};
#endif 