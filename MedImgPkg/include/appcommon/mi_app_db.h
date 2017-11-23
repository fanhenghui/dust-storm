#ifndef MED_IMG_APPCOMMON_MI_APP_DATABASE_H
#define MED_IMG_APPCOMMON_MI_APP_DATABASE_H

#include <string>
#include <vector>
#include "appcommon/mi_app_common_export.h"
#include "io/mi_mysql_db.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export DB : public MySQLDB {
public:
    struct ImgItem {
        std::string series_id;
        std::string study_id;
        std::string patient_name;
        std::string patient_id;
        std::string modality;
        std::string dcm_path;
        std::string preprocess_mask_path;
        std::string annotation_ai_path;
        std::string ai_intermediate_data_path;
        float size_mb;
    };

    struct AnnoItem {
        std::string series_id;
        std::string usr_name;
        std::string annotation_usr_path;
    };

public:
    DB();
    virtual ~DB();

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