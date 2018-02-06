#ifndef MEDIMGIO_DICOM_INFO_H
#define MEDIMGIO_DICOM_INFO_H

#include "io/mi_io_export.h"

MED_IMG_BEGIN_NAMESPACE

// For 
// struct DcmInfo {
//     //patient info
//     int64_t patient_pk;//For retrieve (DB only)
//     std::string patient_id;
//     std::string patient_name;
//     std::string patient_sex;
//     std::string patient_birth_date;//format: YYYYMMDD
    
//     //study info
//     int64_t study_pk;//For retrieve (DB only)
//     std::string study_id;
//     std::string study_uid;
//     std::string accession_no;
//     std::string study_desc;
//     std::string study_date;//format: YYYYMMDD
//     std::string study_time;//format: HHMMSS
//     int num_series;
//     int num_instance_study_related;

//     //series info
//     int64_t series_pk;//For retrieve (DB only)
//     std::string series_id;
//     std::string modality;
//     std::string series_no;
//     std::string institution;
//     std::string series_desc;
//     int num_instance_series_related;
// };

//-------------------------------------------------//
// PACA & DB common
//-------------------------------------------------//

struct PatientInfo {
    int64_t id;//for DB
    std::string patient_id;
    std::string patient_name;
    std::string patient_sex;
    std::string patient_birth_date;//format: YYYYMMDD  //---not query key---//
    std::string md5;

    PatientInfo(): id(-1) {}
};

struct StudyInfo {
    int64_t id;//For-DB //---not query key---//
    
    int patient_fk;//For-DB //---not query key---//

    std::string study_id;
    std::string study_uid;
    std::string study_date;//format: YYYYMMDD, or YYYYMMDD-YYYYMMDD for query  
    std::string study_time;//format: HHMMSS, or HHMMSS-HHMMSS for query
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

struct InstanceInfo {
    std::string sop_class_uid;
    std::string sop_instance_uid;
    std::string file_path;
    int64_t file_size;

    InstanceInfo(): file_size(0) {}
};

enum RoleType {
    CHIEF_PHYSICAN = 1,//主任医师
    ASSOCIATE_CHIEF_PHYSICAN = 2,//副主任医师
    ATTENDING_PHYSICAN = 3,//主治医师
    RESIDENT_PHYSICAN = 4,//住院医师
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
};

enum EvaluationType {
    LUNG_NODULE = 1,
};

struct EvaluationInfo {
    int64_t id;
    int64_t series_fk;
    int eva_type;
    std::string version;
    std::string file_path;//---not query key---//
    int64_t file_size;//---not query key---//

    EvaluationInfo(): id(-1), series_fk(-1), eva_type(-1), file_size(0) {}
};

struct AnnotationInfo {
    int64_t id;
    int64_t series_fk;
    int anno_type;
    int64_t user_id;
    std::string anno_desc;//---not query key---//
    std::string file_path;//---not query key---//
    int64_t file_size;//---not query key---//

    AnnotationInfo(): id(-1), series_fk(-1), anno_type(-1), user_id(-1), file_size(0) {}
};

enum PreprocessType {
    INIT_SEGMENT_MASK = 1,
    LUNG_AI_DATA = 2,
};

struct PreprocessInfo {
    int64_t id;
    int64_t series_fk;
    int prep_type;
    std::string version;
    std::string file_path;
    int64_t file_size;

    PreprocessInfo(): id(-1), series_fk(-1), prep_type(-1) {}
};

MED_IMG_END_NAMESPACE

#endif