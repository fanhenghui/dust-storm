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
    struct ImgItem {
        std::string study_uid;
        std::string series_uid;
        unsigned int study_timestamp;

        std::string modality;
        
        std::string patient_name;
        std::string patient_id;
        std::string patient_sex;
        unsigned int patient_birth_timestamp;
        std::string accession_no;

        int instance_number;
        float size_mb;
        std::string dcm_path;

        std::string preprocess_mask_path;
        std::string annotation_ai_path;
        std::string ai_intermediate_data_path;
    };

    struct EvaItem {
        int series_id;
        std::string lung_eva_version;
        std::string lung_eva_file_path;
        long long lung_eva_file_size;
    };

    struct AnnoItem {
        int series_id;
        int user_id;
        std::string annotation_type;
        std::string annotation_desc;
        std::string annotation_path;
    };

public:
    DB();
    virtual ~DB();
    //add 
    int insert_dcm_series(const DcmInfo& series_info, const std::vector<DcmInstanceInfo>& instance_info, const std::string& user);
    int insert_evaluation(const EvaItem& eva, const )
    //delete 

    //modify

    //query
    void query_user();

    int insert_dcm_item(const DB::ImgItem& item);
    int delete_dcm_item(const std::string& series_id);
    int query_dcm_item(const std::string& series_id, bool& in_db);
    int get_dcm_item(const std::string& series_id, DB::ImgItem& item);

    int get_ai_annotation_item(const std::string& series_id, std::string& annotation_ai_path);
    int get_usr_annotation_items_by_series(const std::string& series_id, std::vector<DB::AnnoItem>& items);
    int get_usr_annotation_items_by_usr(const std::string& usr_name, std::vector<DB::AnnoItem>& items);
    int get_usr_annotation_item(const std::string& series_id, const std::string& usr_name, std::vector<DB::AnnoItem>& items);

    int get_all_dcm_items(std::vector<DB::ImgItem>& items);
    int get_all_usr_annotation_items(std::vector<DB::AnnoItem>& items);

    int update_preprocess_mask(const std::string& series_id, const std::string& preprocess_mask_path);
    int update_ai_annotation(const std::string& series_id, const std::string& annotation_ai_path);
    int update_ai_intermediate_data(const std::string& series_id, const std::string& ai_intermediate_data);
    int update_usr_annotation(const std::string& series_id, const std::string& usr_name, const std::string& annotation_usr_path);

private:

    DISALLOW_COPY_AND_ASSIGN(DB);
};

MED_IMG_END_NAMESPACE

#endif