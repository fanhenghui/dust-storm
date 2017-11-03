#include "mi_app_cache_db.h"

#include "cppconn/driver.h"
#include "cppconn/exception.h"
#include "cppconn/prepared_statement.h"
#include "cppconn/resultset.h"
#include "cppconn/sqlstring.h"
#include "cppconn/statement.h"
#include "mysql_connection.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

const static std::string DCM_TABLE = "dcm_series";

CacheDB::CacheDB() {}

CacheDB::~CacheDB() {
}

int CacheDB::insert_item(const ImgItem& item) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN cache db inset item."; 
    if (!this->is_valid()) {
        MI_APPCOMMON_LOG(MI_ERROR) << "cache db connection invalid.";
        return -1;
    }

    //query    
    bool in_db(false);
    if(-1 == query_item(item.series_id, in_db)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "query failed when insert item.";
        return -1;
    }

    //delete if exit
    if (in_db) {
        if(-1 == delete_item(item.series_id)) {
            MI_APPCOMMON_LOG(MI_ERROR) << "delete item failed when insert the item with the same primary key.";
            return -1;
        }
    }

    // insert new one
    try {
        // insert new item
        std::string sql_str;
        {
            std::stringstream ss;
            ss << "INSERT INTO " << DCM_TABLE << " (series_id, study_id, patient_name, "
               "patient_id, modality, size_mb, path) values (";
            ss << "\'" << item.series_id << "\',";
            ss << "\'" << item.study_id << "\',";
            ss << "\'" << item.patient_name << "\',";
            ss << "\'" << item.patient_id << "\',";
            ss << "\'" << item.modality << "\',";
            ss << "\'" << item.size_mb << "\',";
            ss << "\'" << item.path << "\'";
            ss << ");";
            sql_str = ss.str();
        }
        sql::PreparedStatement* pstmt = _connection->prepareStatement(sql_str.c_str());
        sql::ResultSet* res = pstmt->executeQuery();
        delete pstmt;
        pstmt = nullptr;
        delete res;
        res = nullptr;

    } catch (const sql::SQLException& e) {
        std::stringstream ss;
        ss << "# ERR: " << e.what()
        << " (MySQL error code: " << e.getErrorCode()
        << ", SQLState: " << e.getSQLState() << " )";
        MI_APPCOMMON_LOG(MI_ERROR) << "qurey db when inset item failed with exception: " << ss.str();
        
        //TODO recovery old one if insert failed
        return -1;
    }
    MI_APPCOMMON_LOG(MI_TRACE) << "OUT db query inset item.";
    return 0;
}

int CacheDB::delete_item(const std::string& series_id) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN cache db query delete item."; 
    if (!this->is_valid()) {
        MI_APPCOMMON_LOG(MI_ERROR) << "cache db connection invalid.";
        return -1;
    }

    //query    
    bool in_db(false);
    if(-1 == query_item(series_id, in_db)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "query failed when insert item.";
        return -1;
    }

    if (!in_db) {
        MI_APPCOMMON_LOG(MI_WARNING) << "delete item not exist which series is: " << series_id;        
        return 0;
    }

    //delete
    try {
        std::string sql_str;
        {
            std::stringstream ss;
            ss << "DELETE FROM " << DCM_TABLE << " where series_id=\'" << series_id << "\';";
            sql_str = ss.str();
        }
        sql::PreparedStatement* pstmt = _connection->prepareStatement(sql_str.c_str());
        sql::ResultSet* res = pstmt->executeQuery();
        delete pstmt;
        pstmt = nullptr;
        delete res;
        res = nullptr;
    } catch (const sql::SQLException& e) {
        std::stringstream ss;
        ss << "# ERR: " << e.what()
        << " (MySQL error code: " << e.getErrorCode()
        << ", SQLState: " << e.getSQLState() << " )";
        MI_APPCOMMON_LOG(MI_ERROR) << "qurey db when inset item failed with exception: " << ss.str();
        // TODO recovery DB if delete failed
        return -1;
    }
    MI_APPCOMMON_LOG(MI_TRACE) << "OUT cache db delete item."; 
    return 0;
}

int CacheDB::query_item(const std::string& series_id, bool& in_db) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN cache db query item."; 
    if (!this->is_valid()) {
        MI_APPCOMMON_LOG(MI_ERROR) << "cache db connection invalid.";
        return -1;
    }

    try {
        sql::PreparedStatement* pstmt = nullptr;
        sql::ResultSet* res = nullptr;
        std::string sql_str;
        {
            std::stringstream ss;
            ss << "SELECT * FROM " << DCM_TABLE << " where series_id=\'" << series_id << "\';";
            sql_str = ss.str();
        }
        pstmt = _connection->prepareStatement(sql_str.c_str());
        res = pstmt->executeQuery();
        delete pstmt;
        pstmt = nullptr;

        if (res->next()) {
            delete res;
            res = nullptr;
            in_db = true;
        } else {
            delete res;
            res = nullptr;
            in_db = false;
        }
    } catch (const sql::SQLException& e) {
        std::stringstream ss;
        ss << "# ERR: " << e.what()
        << " (MySQL error code: " << e.getErrorCode()
        << ", SQLState: " << e.getSQLState() << " )";
        MI_APPCOMMON_LOG(MI_ERROR) << "cache db query item failed with exception: " << ss.str();
        return -1;
    }

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT cache db query item."; 
    return 0;
}

int CacheDB::get_item(const std::string& series_id, ImgItem& item) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN cache db get item."; 
    if (!this->is_valid()) {
        MI_APPCOMMON_LOG(MI_ERROR) << "cache db connection invalid.";
        return -1;
    }

    try {
        std::stringstream ss;
        ss << "SELECT * FROM " << DCM_TABLE << " WHERE series_id=\'" << series_id << "\';";
        sql::PreparedStatement* pstmt = _connection->prepareStatement(ss.str().c_str());
        sql::ResultSet* res = pstmt->executeQuery();

        if (res->next()) {
            item.series_id = res->getString("series_id");
            item.study_id = res->getString("study_id");
            item.patient_name = res->getString("patient_name");
            item.patient_id = res->getString("patient_id");
            item.modality = res->getString("modality");
            item.path = res->getString("path");
            item.size_mb = res->getDouble("size_mb");
            delete res;
            res = nullptr;
            delete pstmt;
            pstmt = nullptr;
        } else {
            delete res;
            res = nullptr;
            delete pstmt;
            pstmt = nullptr;
            return -1;
        }
    } catch (const sql::SQLException& e) {
        std::stringstream ss;
        ss << "# ERR: " << e.what()
        << " (MySQL error code: " << e.getErrorCode()
        << ", SQLState: " << e.getSQLState() << " )";
        MI_APPCOMMON_LOG(MI_ERROR) << "cache db get item failed with exception: " << ss.str();
        return -1;
    }

    MI_APPCOMMON_LOG(MI_TRACE) << "Out cache db get item."; 
    return 0;
}

int CacheDB::get_all_items(std::vector<ImgItem>& items) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN cache db get all items."; 
    if (!this->is_valid()) {
        MI_APPCOMMON_LOG(MI_ERROR) << "cache db connection invalid.";
        return -1;
    }

    try {
        std::stringstream ss;
        ss << "SELECT * FROM " << DCM_TABLE;
        sql::PreparedStatement* pstmt = _connection->prepareStatement(ss.str().c_str());
        sql::ResultSet* res = pstmt->executeQuery();

        items.clear();
        std::cout << res->rowsCount() << std::endl;

        while (true) {
            if (res->next()) {
                ImgItem item;
                item.series_id = res->getString("series_id");
                item.study_id = res->getString("study_id");
                item.patient_name = res->getString("patient_name");
                item.patient_id = res->getString("patient_id");
                item.modality = res->getString("modality");
                item.path = res->getString("path");
                item.size_mb = res->getDouble("size_mb");
                items.push_back(item);
            } else {
                break;
            }
        }

        delete res;
        res = nullptr;
        delete pstmt;
        pstmt = nullptr;
    } catch (const sql::SQLException& e) {
        std::stringstream ss;
        ss << "# ERR: " << e.what()
        << " (MySQL error code: " << e.getErrorCode()
        << ", SQLState: " << e.getSQLState() << " )";
        MI_APPCOMMON_LOG(MI_ERROR) << "cache cache db get all items failed with exception: " << ss.str();
        return -1;
    }
    MI_APPCOMMON_LOG(MI_TRACE) << "OUT cache db get all items.";
    return 0;
}

MED_IMG_END_NAMESPACE