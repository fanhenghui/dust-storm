#ifndef MEDIMGAPPCOMMON_MI_APP_DATA_BASE_H
#define MEDIMGAPPCOMMON_MI_APP_DATA_BASE_H

#include <string>
#include <vector>
#include "appcommon/mi_app_common_export.h"

namespace sql {
class Connection;
}

MED_IMG_BEGIN_NAMESPACE

struct ImgItem {
    std::string series_id;
    std::string study_id;
    std::string patient_name;
    std::string patient_id;
    std::string modality;
    std::string path;
};

class AppCommon_Export AppDataBase {
public:
    AppDataBase();
    ~AppDataBase();

    int connect(const std::string& user, const std::string& ip_port,
                const std::string& pwd, const std::string& db_name);
    int disconnect();

    int insert_item(const ImgItem& item);

    int get_series_path(const std::string& series_id , std::string& path ,
                        const std::string& tbl = std::string("img_tbl"));
    int get_all_item(std::vector<ImgItem>& items, const std::string& tbl = std::string("img_tbl"));

private:
    sql::Connection* _connection;

private:
    DISALLOW_COPY_AND_ASSIGN(AppDataBase);
};

MED_IMG_END_NAMESPACE

#endif