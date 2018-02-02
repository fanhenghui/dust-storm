#ifndef MEDIMGIO_DICOM_INFO_H
#define MEDIMGIO_DICOM_INFO_H

#include "io/mi_io_export.h"

MED_IMG_BEGIN_NAMESPACE

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

//-------------------------------------------------//
// PACA & DB common
//-------------------------------------------------//

struct PatientInfo {
    int64_t id;//for DB
    std::string patient_id;
    std::string patient_name;
    std::string patient_sex;
    std::string patient_birth_date;//format: YYYYMMDD
    std::string md5;

    PatientInfo(): id(-1) {}
};

struct StudyInfo {
    int64_t id;//For-DB //---not query key---//
    
    int patient_fk;//For-DB //---not query key---//

    std::string study_id;
    std::string study_uid;
    std::string study_date;//format: YYYYMMDD
    std::string study_time;//format: HHMMSS
    std::string accession_no;
    std::string study_desc;//---not query key---//
    int num_series;//---not query key---//
    int num_instance;//---not query key---//

    StudyInfo():id(-1), patient_fk(-1), num_series(0), num_instance(0) {}
};

struct SeriesInfo {
    int64_t id;//For-DB

    int study_fk;//For-DB
        
    std::string series_uid;
    std::string series_no;
    std::string modality;
    std::string series_desc;//---not query key---//
    std::string institution;
    int num_instance;//---not query key---//

    SeriesInfo():id(-1), study_fk(-1), num_instance(0) {}
};

//-------------------------------------------------//
// DB
//-------------------------------------------------//

struct DcmInstanceInfo {
    std::string sop_class_uid;
    std::string sop_instance_uid;
    std::string file_path;

    DcmInstanceInfo() {}
    DcmInstanceInfo(std::string sop_class_uid_, std::string sop_instance_uid_, std::string file_path_): 
    sop_class_uid(sop_class_uid_), sop_instance_uid(sop_instance_uid_), file_path(file_path_) {}
};

struct RoleInfo {
    int id;
    std::string name;

    RoleInfo(): id(-1) {}
};

struct UserInfo {
    std::string id;
    std::string name;
};

struct EvaluationInfo {
    int64_t series_id;
    int eva_type;
    std::string eva_version;
    std::string eva_file_path;//---not query key---//
    long long eva_file_size;//---not query key---//

    EvaluationInfo(): series_id(-1), eva_type(-1), eva_file_size(0) {}
};

struct AnnotationInfo {
    int64_t series_id;
    int anno_type;
    int64_t user_id;
    std::string anno_desc;//---not query key---//
    std::string anno_file_path;//---not query key---//
    long long anno_file_size;//---not query key---//

    AnnotationInfo(): series_id(-1), anno_type(-1), user_id(-1), anno_file_size(0) {}
};

struct PreprocessInfo {
    int64_t series_id;
    int prep_type;
    std::string prep_version;
    std::string prep_file_path;
    std::string prep_file_size;

    PreprocessInfo(): series_id(-1), prep_type(-1) {}
};

MED_IMG_END_NAMESPACE

#endif