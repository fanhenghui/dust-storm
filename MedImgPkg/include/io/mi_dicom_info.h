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
    std::string study_description;
    std::string patient_id;
    std::string patient_name;
    std::string patient_sex;
    std::string patient_age;
    std::string patient_birth_date;
    std::string modality;
};

MED_IMG_END_NAMESPACE

#endif