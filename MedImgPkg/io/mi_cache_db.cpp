#include "mi_cache_db.h"

#include "cppconn/driver.h"
#include "cppconn/exception.h"
#include "cppconn/prepared_statement.h"
#include "cppconn/resultset.h"
#include "cppconn/sqlstring.h"
#include "cppconn/statement.h"

#include "util/mi_memory_shield.h"

#include "mysql_connection.h"
#include "mi_io_logger.h"

MED_IMG_BEGIN_NAMESPACE

const static std::string SERIES_TABLE = "series";
const static std::string INSTANCE_TABLE = "instance";

#define TRY_CONNECT \
if (!this->try_connect()) {\
    MI_IO_LOG(MI_ERROR) << "db connection invalid.";\
    return -1;\
}

const static int UID_LIMIT = 64;

#define THROW_SQL_EXCEPTION throw std::exception(std::logic_error("sql error"));

CacheDB::CacheDB() {}

CacheDB::~CacheDB() {}

int CacheDB::insert_series(const std::string& series_uid, const std::vector<InstanceInfo>& instance_infos) {
    //verify
    if (series_uid.empty() || series_uid.size() > UID_LIMIT) {
        MI_IO_LOG(MI_ERROR) << "cache insert series failed: invalid series_uid: " << series_uid;
        return -1;
    }

    if (instance_infos.empty()) {
        MI_IO_LOG(MI_ERROR) << "cache insert series failed: emtpy instance.";
        return -1;
    }

    TRY_CONNECT
    int err = 0;
    _connection->setAutoCommit(false);
    sql::Savepoint* save_point = _connection->setSavepoint("cache_insert_series");
    StructShield<sql::Savepoint> shield(save_point);
    try {
        //---------------------------//
        //delete old series
        //---------------------------//
        if (series_uid.size() > UID_LIMIT) {
            throw std::exception(std::logic_error("invalid series uid."));
        }

        std::stringstream sql_select_series;
        sql_select_series << "SELECT id FROM " << SERIES_TABLE
        << " WHERE series_uid=\'" << series_uid << "\';";

        sql::ResultSet* res_select_series = nullptr;
        err = query(sql_select_series.str(), res_select_series);
        StructShield<sql::ResultSet> shield_select_series(res_select_series);
        if (0 != err) {
            throw std::exception(std::logic_error("query series failed."));
        } else if (res_select_series->next()) {
            if (0!= delete_series(res_select_series->getInt64("id"), false)) {
                throw std::exception(std::logic_error("delete series failed."));    
            }

            MI_IO_LOG(MI_DEBUG) << "cache insert dcm series: delete old series done";
        }

        //---------------------------//
        //insert into series
        //---------------------------//
        std::stringstream sql;
        sql << "INSERT INTO " << SERIES_TABLE << "(series_uid) VALUES(\'" << series_uid <<"\');"; 

        sql::ResultSet* res_insert_series = nullptr;
        err = this->query(sql.str(), res_insert_series);
        StructShield<sql::ResultSet> shield_insert_series(res_insert_series);
        if(0 != err) {
            throw std::exception(std::logic_error("insert series table sql error."));
        }

        sql::ResultSet* res_select_series2 = nullptr;
        err = query(sql_select_series.str(), res_select_series2);
        StructShield<sql::ResultSet> shield_select_series2(res_select_series2);
        if (0 != err) {
            throw std::exception(std::logic_error("query series failed 2."));
        }

        int series_pk = -1;
        if (res_select_series2->next()) {
            series_pk = res_select_series2->getInt64("id");
        }
        if (series_pk < 1) {
            throw std::exception(std::logic_error("invalid series pk."));
        }


        //---------------------------//
        //insert into instance
        //---------------------------//
        for (auto it = instance_infos.begin(); it != instance_infos.end(); ++it) {
            const InstanceInfo &info = *it;
            //verfiy
            if (info.sop_instance_uid.empty()) {
                throw std::exception(std::logic_error("invalid instance info: emtpy sop instance uid."));
            }
            if (info.file_path.empty()) {
                throw std::exception(std::logic_error("invalid instance info: emtpy file path."));
                return -1;
            }
            if (info.file_size <=0) {
                throw std::exception(std::logic_error("invalid instance info: emtpy file size."));
                return -1;
            }

            std::stringstream sql;
            sql << "INSERT INTO " << INSTANCE_TABLE
            << "(series_fk, sop_instance_uid, file_path, file_size) VALUES("
            << "\'" << series_pk << "\',"
            << "\'" << info.sop_instance_uid << "\',"
            << "\'" << info.file_path << "\',"
            << "\'" << info.file_size << "\'"
            << ");";

            sql::ResultSet* res = nullptr;
            int err = this->query(sql.str(), res);
            StructShield<sql::ResultSet> shield(res);
            if(0 != err) {
                THROW_SQL_EXCEPTION
            }
        }

    } catch (const std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "cache insert seriess failed: " << e.what();
        _connection->rollback(save_point);
        return -1;
    }

    _connection->commit();

    return 0;
}

int CacheDB::delete_series(int series_pk, bool transcation) {
    //TODO
    return 0;
}

int CacheDB::query_series_instance(const std::string& series_uid, std::vector<InstanceInfo>* instance_infos) {
    //verify
    if (series_uid.empty() || series_uid.size() > UID_LIMIT) {
        MI_IO_LOG(MI_ERROR) << "cache query series instance failed: invalid series_uid: " << series_uid;
        return -1;
    }

    try {
        TRY_CONNECT

        std::stringstream sql;
        sql << "SELECT id FROM " << SERIES_TABLE
        << " WHERE series_uid=\'" << series_uid << "\';";

        sql::ResultSet* res_series = nullptr;
        int err = this->query(sql.str(), res_series);
        StructShield<sql::ResultSet> shield_series(res_series);
        if (0 != err) {
            throw std::exception(std::logic_error("query series failed."));
        }

        int series_pk = -1;
        if (res_series->next()) {
            series_pk = res_series->getInt64("id");
        }
        if (series_pk < 1) {
            throw std::exception(std::logic_error("invalid series pk."));
        }

        
        sql.str("");
        sql << "SELECT file_path, file_size FROM " 
        << INSTANCE_TABLE << " WHERE series_fk=" << series_pk << ";";

        sql::ResultSet* res = nullptr;
        err = this->query(sql.str(), res);
        StructShield<sql::ResultSet> shield(res);

        if (0 != err) {
            throw std::exception(std::logic_error("query instance failed."));
        }

        instance_infos->clear();
        while(res->next()) {
            instance_infos->push_back(InstanceInfo());
            InstanceInfo& info = (*instance_infos)[instance_infos->size() - 1];
            info.file_path = res->getString("file_path").asStdString();
            info.file_size = res->getInt64("file_size");
        }
    } catch (std::exception& e) {
        MI_IO_LOG(MI_ERROR) << "query series instance failed: " << e.what();
        return -1;
    }

    return 0;
}

MED_IMG_END_NAMESPACE