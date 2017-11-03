#ifndef MED_IMG_APPCOMMON_MI_APP_CACHE_DATABASE_H
#define MED_IMG_APPCOMMON_MI_APP_CACHE_DATABASE_H

#include <string>
#include <vector>
#include "appcommon/mi_app_common_export.h"
#include "io/mi_mysql_db.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export CacheDB : public MySQLDB {
public:
    struct ImgItem {
        std::string series_id;
        std::string study_id;
        std::string patient_name;
        std::string patient_id;
        std::string modality;
        std::string path;
        int size_mb;
    };

public:
    CacheDB();
    virtual ~CacheDB();

    int insert_item(const CacheDB::ImgItem& item);
    int delete_item(const std::string& series_id);
    int get_item(const std::string& series_id, CacheDB::ImgItem& item);
    int query_item(const std::string& series_id, bool &in_db);
 
    int get_all_items(std::vector<CacheDB::ImgItem>& items);

private:
    DISALLOW_COPY_AND_ASSIGN(CacheDB);
};

MED_IMG_END_NAMESPACE

#endif