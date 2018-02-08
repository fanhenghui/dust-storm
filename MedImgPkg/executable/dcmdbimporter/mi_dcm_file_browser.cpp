#include "mi_dcm_file_browser.h"

#include "util/mi_file_util.h"
#include "io/mi_dicom_loader.h"
#include "log/mi_logger.h"

namespace medical_imaging {

int DcmFileBrowser::browse(const std::string& direction) {
    std::vector<std::string> files;
    std::set<std::string> postfix;
    postfix.insert(".dcm");
    if (0 != FileUtil::get_all_file_recursion(direction, postfix, files)) {
        MI_LOG(MI_ERROR) << "get direction: " << direction << " failed.";
        return -1;
    }

    _study_infos.clear();
    _series_infos.clear();
    _patient_infos.clear();

    //-----------------------------//
    // extract all series
    //-----------------------------//
    DICOMLoader loader;
    std::string series_uid;
    std::string study_uid;
    for (auto it = files.begin(); it != files.end(); ++it) {
        const std::string& file = *it;
        if (IO_SUCCESS != loader.check_series_uid(file, study_uid, series_uid)) {
            MI_LOG(MI_WARNING) << "check DICOM file: " << file << " failed.";
            continue;
        }
        PatientInfo patient_info;
        StudyInfo study_info;
        SeriesInfo series_info;
        InstanceInfo instance_info;
        instance_info.file_path = file;

        if (0 != FileUtil::get_file_size(file, instance_info.file_size)) {
            MI_LOG(MI_WARNING) << "get file: " << file << " size failed.";
            continue;
        }

        if( IO_SUCCESS != loader.get_dicom_info(file, 
            patient_info, study_info, series_info, instance_info)) {
            MI_LOG(MI_WARNING) << "get DICOM info from file: " << file << " failed.";
            continue;
        }

        auto it_se = _series_infos.find(series_uid);
        if (it_se == _series_infos.end()) {
            //get study series patient info
            _series_infos.insert(std::make_pair(series_uid, series_info));
            _patient_infos.insert(std::make_pair(series_uid, patient_info));
            _study_infos.insert(std::make_pair(series_uid, study_info));
            _instance_infos.insert(std::make_pair(series_uid, std::vector<InstanceInfo>())); 
        }

        _series_infos[series_uid].num_instance += 1;
        _instance_infos[series_uid].push_back(instance_info);
    }

    return 0;
}

};