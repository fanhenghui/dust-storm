#ifndef MEDIMGIO_DICOM_INFO_H
#define MEDIMGIO_DICOM_INFO_H

#include "io/mi_io_export.h"

MED_IMG_BEGIN_NAMESPACE

//basic DICOM tag collection for query and filter
struct DcmInfo {
    std::string study_id;
    std::string series_id;
    std::string study_date;
    std::string study_time;
    std::string patient_id;
    std::string patient_name;
    std::string patient_sex;
    std::string patient_birth_date;
    std::string modality;
    std::string accession_number;
    std::string series_no;
    std::string institution;
    std::string series_desc;
    std::string study_desc;
    int number_of_instance;
    int number_of_series;//just for study level query
};

struct DcmInstanceInfo {
    std::string sop_class_uid;
    std::string sop_instance_uid;
    std::string file_path;

    DcmInstanceInfo() {}
    DcmInstanceInfo(std::string sop_class_uid_, std::string sop_instance_uid_, std::string file_path_): 
    sop_class_uid(sop_class_uid_), sop_instance_uid(sop_instance_uid_), file_path(file_path_) {}
};

struct QueryKey {
    std::string study_uid;
    std::string series_uid;
    std::string study_date;
    std::string study_time;
    std::string patient_id;
    std::string patient_name;
    std::string modality;
    std::string accession_number;
    std::string patient_sex;
};

MED_IMG_END_NAMESPACE

#endif