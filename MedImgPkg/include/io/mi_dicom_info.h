#ifndef MEDIMGIO_DICOM_INFO_H
#define MEDIMGIO_DICOM_INFO_H

#include "io/mi_io_export.h"

MED_IMG_BEGIN_NAMESPACE

//basic DICOM tag collection for query and filter
enum QueryLevel {
    PATIENT = 0,
    STUDY = 1,
    SERIES = 2,
};

struct DcmInfo {
    std::string study_id;
    std::string series_id;
    std::string study_date;//format: YYYYMMDD
    std::string study_time;//format: HHMMSS
    std::string patient_id;
    std::string patient_name;
    std::string patient_sex;
    std::string patient_birth_date;
    std::string modality;
    std::string accession_no;
    std::string series_no;
    std::string institution;
    std::string series_desc;
    std::string study_desc;
    int number_of_instance;
    int number_of_series;//just for study level query
};

struct PatientInfo {
    int id;//for DB
    std::string patient_id;
    std::string patient_name;
    std::string patient_sex;
    std::string patient_birth_date;
};

struct StudyInfo {
    int id;//for DB
    
    int patient_fk;//for DB
    std::string patient_name;//for PACS

    std::string study_id;
    std::string study_uid;
    std::string study_date;//format: YYYYMMDD
    std::string study_time;//format: HHMMSS
    std::string accession_no;
    std::string study_desc;
    int num_series;
    int num_instance;
};

struct SeriesInfo {
    int id;

    int study_fk;
    std::string study_uid;
        
    std::string series_uid;
    std::string series_no;
    std::string modality;
    std::string series_desc;
    std::string institution;
    int num_instance;
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
    std::string study_date;//format: YYYYMMDD
    std::string study_time;//format: HHMMSS
    std::string patient_id;
    std::string patient_name;
    std::string modality;
    std::string accession_no;
    std::string patient_sex;
    std::string patient_birth_date;
};

struct RoleInfo {
    int id;
    std::string name;
};

struct UserInfo {
    std::string id;
    std::string name;
};

struct EvaluationInfo {
    int series_id;
    int eva_type;
    std::string eva_version;
    std::string eva_file_path;
    long long eva_file_size;
};

struct AnnotationInfo {
    int series_id;
    int anno_type;
    int user_id;
    std::string anno_desc;
    std::string anno_file_path;
    long long anno_file_size;
};

struct PreprocessInfo {
    int series_id;
    int prep_type;
    std::string prep_version;
    std::string prep_file_path;
    std::string prep_file_size;
};

MED_IMG_END_NAMESPACE

#endif