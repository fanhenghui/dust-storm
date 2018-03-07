#ifndef MEDIMG_IO_MI_DB_H
#define MEDIMG_IO_MI_DB_H

#include <string>
#include <vector>
#include "io/mi_mysql_db.h"
#include "io/mi_dicom_info.h"

MED_IMG_BEGIN_NAMESPACE

//---------------------------------------------------//
// Instnace: 
//     DICOM series(.dcm)
//     Evaluation(.csv)
//     Preprocess Data(.npy .rle); 
//     Annotation(.csv)
//---------------------------------------------------//
class IO_Export DB : public MySQLDB {
public:
    DB();
    virtual ~DB();
    
    //--------------------//
    // insert 
    //--------------------//
    int insert_series(PatientInfo& patient_info, StudyInfo& study_info, SeriesInfo& series_info,
         const std::vector<InstanceInfo>& instance_info);

    int insert_evaluation(const EvaluationInfo& eva_info);
    int insert_annotation(const AnnotationInfo& anno_info);
    int insert_preprocess(const PreprocessInfo& prep_info);

    //--------------------//
    // update
    //--------------------//
    int update_patient(PatientInfo& patient_info);
    int update_study(StudyInfo& study_info);
    int update_series(SeriesInfo& series_info);

    int update_evaluation(const EvaluationInfo& eva_info);
    int update_annotation(const AnnotationInfo& anno_info);
    int update_preprocess(const PreprocessInfo& prep_info);

    //--------------------//
    // deelte
    //--------------------//
    int delete_series(int series_pk, bool transcation = true);
    int delete_evaluation(int eva_pk);
    int delete_annotation(int anno_pk);
    int delete_preprocess(int prep_pk);

    //--------------------//
    // query
    //--------------------//
    //DICOM query (corresponding to query level)
    int query_patient(const PatientInfo& key, std::vector<PatientInfo>* patient_infos);
    int query_study(const PatientInfo& patient_key, const StudyInfo& study_key, 
        std::vector<PatientInfo>* patient_infos, std::vector<StudyInfo>* study_infos);
    int query_series(const PatientInfo& patient_key, const StudyInfo& study_key, const SeriesInfo& series_key, 
        std::vector<PatientInfo>* patient_infos, std::vector<StudyInfo>* study_infos, std::vector<SeriesInfo>* series_infos);

    //query series instance for data transform(DB->CACHE or DB->APP)
    int query_series_instance(int64_t series_pk, std::vector<InstanceInfo>* instance_infos);
    int query_series_instance(int64_t series_pk, std::vector<std::string>* instance_file_paths);

    //query series uid
    int query_series_uid(int64_t series_pk, std::vector<std::string>* series_uid);

    int query_user(const UserInfo& key, std::vector<UserInfo>* user_infos);
    int query_evaluation(const EvaluationInfo& key, std::vector<EvaluationInfo>* eva_infos);
    int query_annotation(const AnnotationInfo& key, std::vector<AnnotationInfo>* anno_infos);
    int query_preprocess(const PreprocessInfo& key, std::vector<PreprocessInfo>* prep_infos);

    //--------------------//
    // count
    //--------------------//
    int count_patient(int64_t& num);
    int count_study(int64_t& num);
    int count_series(int64_t& num);
    int count_instance(int64_t& num);
    
    int count_user(int64_t& num);
    int count_evaluation(int64_t& num);
    int count_annotation(int64_t& num);
    int count_preprocess(int64_t& num);
    
    //--------------------//
    // sum&statistic
    //--------------------//
    // TODO

private://FOR test , set to private later
    int insert_patient(PatientInfo& patient_info);
    int insert_study(StudyInfo& study_info);
    int insert_series(SeriesInfo& series_info);
    int insert_instance(int64_t series_fk, const std::vector<InstanceInfo>& instance_info);
private:
    int verify_evaluation_info(const EvaluationInfo& info);
    int verify_annotation_info(const AnnotationInfo& info);
    int verify_preprocess_info(const PreprocessInfo& info);
    int verfiy_study_info(StudyInfo& info);
    int verfiy_series_info(const SeriesInfo& info);

    bool patient_key_valid(const PatientInfo& key);
    bool study_key_valid(const StudyInfo& key);
    bool series_key_valid(const SeriesInfo& key);

    int query_study(int64_t patient_fk, const StudyInfo& study_key, std::vector<StudyInfo>* study_infos);
    int query_series(int64_t study_fk, const SeriesInfo& series_key, std::vector<SeriesInfo>* series_infos);

private:
    DISALLOW_COPY_AND_ASSIGN(DB);
};

MED_IMG_END_NAMESPACE

#endif