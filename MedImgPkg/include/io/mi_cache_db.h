#ifndef MEDIMG_IO_MI_CACHE_DB_H
#define MEDIMG_IO_MI_CACHE_DB_H

#include <string>
#include <vector>
#include "io/mi_mysql_db.h"
#include "io/mi_dicom_info.h"

MED_IMG_BEGIN_NAMESPACE

class IO_Export CacheDB : public MySQLDB {
public:
    CacheDB();
    virtual ~CacheDB();

    int insert_series(const std::string& series_uid, const std::vector<InstanceInfo>& instance_info);

    int delete_series(int series_pk, bool transcation = true);

    int query_series_instance(const std::string& series_uid, std::vector<InstanceInfo>* instance_infos);
    int query_series_instance(const std::string& series_uid, std::vector<std::string>* instance_file_paths);

private:
    DISALLOW_COPY_AND_ASSIGN(CacheDB);
};

MED_IMG_END_NAMESPACE

#endif