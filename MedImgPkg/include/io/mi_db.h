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
    
    //insert 
    int insert_dcm_series(StudyInfo& study_info, SeriesInfo& series_info, PatientInfo& patient_info, UserInfo& user_info, 
         const std::vector<DcmInstanceInfo>& instance_info);

    int insert_evaluation(const EvaluationInfo& eva_info);
    int insert_annotation(const AnnotationInfo& anno_info);
    int insert_preprocess(const PreprocessInfo& prep_info);

    //update
    int update_patient(PatientInfo& patient_info);
    int update_study(StudyInfo& study_info);
    int update_series(SeriesInfo& series_info);

    int update_evaluation(const EvaluationInfo& eva_info);
    int update_annotation(const AnnotationInfo& anno_info);
    int update_preprocess(const PreprocessInfo& prep_info);

    //delete
    int delete_dcm_series(int series_fk, bool transcation = true);
    int delete_evaluation(int eva_id);
    int delete_annotation(int anno_id);
    int delete_preprocess(int prep_id);

    //DICOM query
    int query_patient(const PatientInfo& key, std::vector<PatientInfo>* patient_infos);
    int query_study(const PatientInfo& patient_key, const StudyInfo& study_key, 
        std::vector<PatientInfo>* patient_infos, std::vector<StudyInfo>* study_infos);
    int query_series(const PatientInfo& patient_key, const StudyInfo& study_key, const SeriesInfo& series_key, 
        std::vector<PatientInfo>* patient_infos, std::vector<StudyInfo>* study_infos, std::vector<StudyInfo>* series_infos);

    //add query series by series pk(for app loading)

    //for read DICOM file    
    int query_instance(int series_fk, std::vector<DcmInstanceInfo>* instance_infos);

    int query_user(const UserInfo& key, std::vector<UserInfo>* user_infos);
    int query_evaluation(const EvaluationInfo& key, std::vector<EvaluationInfo>* eva_infos);
    int query_annotation(const AnnotationInfo& key, std::vector<AnnotationInfo>* anno_infos);
    int query_preprocess(const PreprocessInfo& key, std::vector<PreprocessInfo>* prep_infos);
    
    

    // int insert_dcm_item(const DB::ImgItem& item);
    // int delete_dcm_item(const std::string& series_fk);
    // int query_dcm_item(const std::string& series_fk, bool& in_db);
    // int get_dcm_item(const std::string& series_fk, DB::ImgItem& item);

    // int get_ai_annotation_item(const std::string& series_fk, std::string& annotation_ai_path);
    // int get_usr_annotation_items_by_series(const std::string& series_fk, std::vector<DB::AnnotationInfo>& items);
    // int get_usr_annotation_items_by_usr(const std::string& usr_name, std::vector<DB::AnnotationInfo>& items);
    // int get_usr_annotation_item(const std::string& series_fk, const std::string& usr_name, std::vector<DB::AnnotationInfo>& items);

    // int get_all_dcm_items(std::vector<DB::ImgItem>& items);
    // int get_all_usr_annotation_items(std::vector<DB::AnnotationInfo>& items);

    // int update_preprocess_mask(const std::string& series_fk, const std::string& preprocess_mask_path);
    // int update_ai_annotation(const std::string& series_fk, const std::string& annotation_ai_path);
    // int update_ai_intermediate_data(const std::string& series_fk, const std::string& ai_intermediate_data);
    // int update_usr_annotation(const std::string& series_fk, const std::string& usr_name, const std::string& annotation_usr_path);

public://FOR test , set to private later
    int insert_patient(PatientInfo& patient_info);
    int insert_study(StudyInfo& study_info);
    int insert_series(SeriesInfo& series_info);
    int insert_instance(const std::string& user_fk, int64_t series_fk, const std::vector<DcmInstanceInfo>& instance_info);
private:
    int verify_evaluation_info(const EvaluationInfo& info);
    int verify_annotation_info(const AnnotationInfo& info);
    int verify_preprocess_info(const PreprocessInfo& info);
    int verfiy_study_info(const StudyInfo& info);
    int verfiy_series_info(const SeriesInfo& info);

    bool patient_key_valid(const PatientInfo& key);
    bool study_key_valid(const StudyInfo& key);

    int query_study(int64_t patient_fk, const StudyInfo& study_key, std::vector<StudyInfo>* study_infos);

private:
    DISALLOW_COPY_AND_ASSIGN(DB);
};

MED_IMG_END_NAMESPACE

#endif