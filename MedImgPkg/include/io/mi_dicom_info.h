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
    int64_t file_size;

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
    int role_fk;
    std::string name;

    UserInfo(): role_fk(-1) {}
    
    UserInfo(const std::string& id_, int role_fk_, const std::string& name_):
    id(id_), role_fk(role_fk_), name(name_) {}
};

struct EvaluationInfo {
    int64_t id;
    int64_t series_fk;
    int eva_type;
    std::string eva_version;
    std::string eva_file_path;//---not query key---//
    int64_t eva_file_size;//---not query key---//

    EvaluationInfo(): id(-1), series_fk(-1), eva_type(-1), eva_file_size(0) {}

    EvaluationInfo(int64_t id_, int64_t series_id_, int eva_type_, 
        const std::string& eva_version, const std::string& eva_file_path_, int64_t eva_file_size_): 
        id(id_), series_fk(series_id_), eva_type(eva_type_), eva_file_path(eva_file_path_),
        eva_file_size(eva_file_size_) {}
};

struct AnnotationInfo {
    int64_t id;
    int64_t series_fk;
    int anno_type;
    int64_t user_id;
    std::string anno_desc;//---not query key---//
    std::string anno_file_path;//---not query key---//
    int64_t anno_file_size;//---not query key---//

    AnnotationInfo(): id(-1), series_fk(-1), anno_type(-1), user_id(-1), anno_file_size(0) {}

    AnnotationInfo(int64_t id_, int64_t series_id_, int anno_type_, 
        int64_t user_id_, const std::string& anno_desc_, const std::string& anno_file_path_, int64_t anno_file_size_): 
        id(id_), series_fk(series_id_), anno_type(anno_type_), user_id(user_id_), 
        anno_desc(anno_desc_), anno_file_path(anno_file_path_), anno_file_size(anno_file_size_) {}
};

struct PreprocessInfo {
    int64_t id;
    int64_t series_fk;
    int prep_type;
    std::string prep_version;
    std::string prep_file_path;
    int64_t prep_file_size;

    PreprocessInfo(): id(-1), series_fk(-1), prep_type(-1) {}
};

MED_IMG_END_NAMESPACE

#endif