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
    int insert_patient(PatientInfo& patient_info);
    int insert_evaluation(const EvaluationInfo& eva);
    int insert_annotation(const AnnotationInfo& anno);
    int insert_preprocess(const PreprocessInfo& prep);

    //delete
    int delete_dcm_series(int series_id);
    int delete_evaluation(int eva_id);
    int delete_annotation(int anno_id);
    int delete_preprocess(int prep_id);

    //modify
    int modify_evaluation(int eva_id, const EvaluationInfo& eva);
    int modify_annotation(int anno_id, const AnnotationInfo& anno);
    int modify_preprocess(int prep_id, const PreprocessInfo& prep);

    //query
    //int query_dicom(std::vector<DcmInfo>& dcm_infos, const QueryKey& key, QueryLevel query_level);
    int query_patient(const PatientInfo& key, std::vector<PatientInfo>* patient_infos);
    int query_study(const PatientInfo& patient_key, const StudyInfo& study_key, std::vector<PatientInfo>* patient_infos, std::vector<StudyInfo>* study_infos);
    int query_series(const PatientInfo& patient_key, const StudyInfo& study_key, const SeriesInfo& series_key, std::vector<PatientInfo>* patient_infos, std::vector<StudyInfo>* study_infos, std::vector<StudyInfo>* series_infos);
    int query_instance(int series_id, std::vector<DcmInstanceInfo>* instance_infos);
    
    int query_user(const std::string& user_name, int role);
    int query_evaluation(int series_id, int eva_type);
    int query_preprocess(int series_id, int prep_type);
    int query_annotation(int series_id, int anno_type, const std::string& user_id);
    

    // int insert_dcm_item(const DB::ImgItem& item);
    // int delete_dcm_item(const std::string& series_id);
    // int query_dcm_item(const std::string& series_id, bool& in_db);
    // int get_dcm_item(const std::string& series_id, DB::ImgItem& item);

    // int get_ai_annotation_item(const std::string& series_id, std::string& annotation_ai_path);
    // int get_usr_annotation_items_by_series(const std::string& series_id, std::vector<DB::AnnotationInfo>& items);
    // int get_usr_annotation_items_by_usr(const std::string& usr_name, std::vector<DB::AnnotationInfo>& items);
    // int get_usr_annotation_item(const std::string& series_id, const std::string& usr_name, std::vector<DB::AnnotationInfo>& items);

    // int get_all_dcm_items(std::vector<DB::ImgItem>& items);
    // int get_all_usr_annotation_items(std::vector<DB::AnnotationInfo>& items);

    // int update_preprocess_mask(const std::string& series_id, const std::string& preprocess_mask_path);
    // int update_ai_annotation(const std::string& series_id, const std::string& annotation_ai_path);
    // int update_ai_intermediate_data(const std::string& series_id, const std::string& ai_intermediate_data);
    // int update_usr_annotation(const std::string& series_id, const std::string& usr_name, const std::string& annotation_usr_path);

private:

    DISALLOW_COPY_AND_ASSIGN(DB);
};

MED_IMG_END_NAMESPACE

#endif