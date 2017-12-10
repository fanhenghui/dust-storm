#include "mi_db.h"

#include "cppconn/driver.h"
#include "cppconn/exception.h"
#include "cppconn/prepared_statement.h"
#include "cppconn/resultset.h"
#include "cppconn/sqlstring.h"
#include "cppconn/statement.h"
#include "mysql_connection.h"
#include "mi_io_logger.h"

MED_IMG_BEGIN_NAMESPACE

const static std::string DCM_TABLE = "dcm_series";
const static std::string ANNO_TABLE = "annotations";

DB::DB() {}

DB::~DB() {
}

int DB::insert_dcm_item(const ImgItem& item) {
    MI_IO_LOG(MI_TRACE) << "IN db inset item."; 
    if (!this->is_valid()) {
        MI_IO_LOG(MI_ERROR) << "db connection invalid.";
        return -1;
    }

    //query    
    bool in_db(false);
    if(-1 == query_dcm_item(item.series_id, in_db)) {
        MI_IO_LOG(MI_ERROR) << "query failed when insert item.";
        return -1;
    }

    //delete if exit
    if (in_db) {
        if(-1 == delete_dcm_item(item.series_id)) {
            MI_IO_LOG(MI_ERROR) << "delete item failed when insert the item with the same primary key.";
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
               "patient_id, modality, size_mb, dcm_path) values (";
            ss << "\'" << item.series_id << "\',";
            ss << "\'" << item.study_id << "\',";
            ss << "\'" << item.patient_name << "\',";
            ss << "\'" << item.patient_id << "\',";
            ss << "\'" << item.modality << "\',";
            ss << "\'" << item.size_mb << "\',";
            ss << "\'" << item.dcm_path << "\'";
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
        MI_IO_LOG(MI_ERROR) << "qurey db when inset item failed with exception: "
        << this->get_sql_exception_info_i(&e);
        
        //TODO recovery old one if insert failed
        return -1;
    }
    MI_IO_LOG(MI_TRACE) << "OUT db query inset item."; 
    return 0;
}

int DB::delete_dcm_item(const std::string& series_id) {
    MI_IO_LOG(MI_TRACE) << "IN db query delete item."; 
    if (!this->is_valid()) {
        MI_IO_LOG(MI_ERROR) << "db connection invalid.";
        return -1;
    }

    //query    
    bool in_db(false);
    if(-1 == query_dcm_item(series_id, in_db)) {
        MI_IO_LOG(MI_ERROR) << "query failed when insert item.";
        return -1;
    }

    if (!in_db) {
        MI_IO_LOG(MI_WARNING) << "delete item not exist which series is: " << series_id;        
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
        MI_IO_LOG(MI_ERROR) << "qurey db when inset item failed with exception: "
        << this->get_sql_exception_info_i(&e);
        // TODO recovery DB if delete failed
        return -1;
    }
    MI_IO_LOG(MI_TRACE) << "OUT db delete item."; 
    return 0;
}

int DB::query_dcm_item(const std::string& series_id, bool& in_db) {
    MI_IO_LOG(MI_TRACE) << "IN db query item."; 
    if (!this->is_valid()) {
        MI_IO_LOG(MI_ERROR) << "db connection invalid.";
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
        MI_IO_LOG(MI_ERROR) << "db query item failed with exception: "
        << this->get_sql_exception_info_i(&e);
        return -1;
    }
    MI_IO_LOG(MI_TRACE) << "OUT db query item.";
    return 0;    
}

int DB::get_dcm_item(const std::string& series_id, ImgItem& item) {
    MI_IO_LOG(MI_TRACE) << "IN db get item."; 
    if (!this->is_valid()) {
        MI_IO_LOG(MI_ERROR) << "db connection invalid.";
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
            item.dcm_path = res->getString("dcm_path");
            item.preprocess_mask_path = res->getString("preprocess_mask_path");
            item.annotation_ai_path = res->getString("annotation_ai_path");
            item.size_mb = res->getInt("size_mb");
            item.ai_intermediate_data_path = res->getString("ai_intermediate_data_path");
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
        MI_IO_LOG(MI_ERROR) << "db get item failed with exception: "
        << this->get_sql_exception_info_i(&e);
        return -1;
    }

    MI_IO_LOG(MI_TRACE) << "Out db get item."; 
    return 0; 
}

int DB::get_ai_annotation_item(const std::string& series_id, std::string& annotation_ai_path) {
    MI_IO_LOG(MI_TRACE) << "IN db get AI annotation item."; 
    if (!this->is_valid()) {
        MI_IO_LOG(MI_ERROR) << "db connection invalid.";
        return -1;
    }

    try {
        std::stringstream ss;
        ss << "SELECT * FROM " << DCM_TABLE << " WHERE series_id=\'" << series_id << "\';";
        sql::PreparedStatement* pstmt = _connection->prepareStatement(ss.str().c_str());
        sql::ResultSet* res = pstmt->executeQuery();

        if (res->next()) {
            annotation_ai_path = res->getString("annotation_ai_path");
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
        MI_IO_LOG(MI_ERROR) << "db get item failed with exception: "
        << this->get_sql_exception_info_i(&e);
        return -1;
    }

    MI_IO_LOG(MI_TRACE) << "Out db get AI annotation item."; 
    return 0; 
}

int DB::get_usr_annotation_items_by_series(const std::string& series_id, std::vector<AnnoItem>& items) {
    MI_IO_LOG(MI_TRACE) << "IN db get usr annotation items by series id.";
    if (!this->is_valid()) {
        MI_IO_LOG(MI_ERROR) << "db connection invalid.";
        return -1;
    }

    try {
        std::stringstream ss;
        ss << "SELECT * FROM " << ANNO_TABLE << " WHERE series_id=\'" << series_id << "\'";
        sql::PreparedStatement* pstmt = _connection->prepareStatement(ss.str().c_str());
        sql::ResultSet* res = pstmt->executeQuery();

        items.clear();
        while(true) {
            if(res->next()) {
                AnnoItem item;
                item.series_id = series_id;
                item.usr_name = res->getString("usr_name");
                item.annotation_usr_path = res->getString("annotation_usr_path");
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
        MI_IO_LOG(MI_ERROR) << "db get annotation items by series id failed with exception: "
        << this->get_sql_exception_info_i(&e);
        return -1;
    }
    MI_IO_LOG(MI_TRACE) << "OUT db get usr annotation items by series id.";
    return 0;
}

int DB::get_usr_annotation_items_by_usr(const std::string& usr_name, std::vector<AnnoItem>& items) {
    MI_IO_LOG(MI_TRACE) << "IN db get usr annotation items by usr.";
    if (!this->is_valid()) {
        MI_IO_LOG(MI_ERROR) << "db connection invalid.";
        return -1;
    }

    try {
        std::stringstream ss;
        ss << "SELECT * FROM " << ANNO_TABLE << " WHERE usr_name=\'" << usr_name << "\'";
        sql::PreparedStatement* pstmt = _connection->prepareStatement(ss.str().c_str());
        sql::ResultSet* res = pstmt->executeQuery();

        items.clear();
        while(true) {
            if(res->next()) {
                AnnoItem item;
                item.series_id = res->getString("series_id");
                item.usr_name = usr_name;
                item.annotation_usr_path = res->getString("annotation_usr_path");
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
        MI_IO_LOG(MI_ERROR) << "db get annotation items by usr name failed with exception: "
        << this->get_sql_exception_info_i(&e);
        return -1;
    }
    MI_IO_LOG(MI_TRACE) << "OUT db get usr annotation items by usr.";
    return 0;
}

int DB::get_usr_annotation_item(const std::string& series_id, const std::string& usr_name, std::vector<DB::AnnoItem>& items) {
    MI_IO_LOG(MI_TRACE) << "IN db get usr annotation item.";
    if (!this->is_valid()) {
        MI_IO_LOG(MI_ERROR) << "db connection invalid.";
        return -1;
    }

    try {
        std::stringstream ss;
        ss << "SELECT * FROM " << ANNO_TABLE << " WHERE usr_name=\'" << usr_name << "\'" 
        << " and series_id=\'" << series_id << "\'";
        sql::PreparedStatement* pstmt = _connection->prepareStatement(ss.str().c_str());
        sql::ResultSet* res = pstmt->executeQuery();

        items.clear();
        while(true) {
            if(res->next()) {
                AnnoItem item;
                item.series_id = series_id;
                item.usr_name = usr_name;
                item.annotation_usr_path = res->getString("annotation_usr_path");
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
        MI_IO_LOG(MI_ERROR) << "db get annotation item failed with exception: "
        << this->get_sql_exception_info_i(&e);
        return -1;
    }
    MI_IO_LOG(MI_TRACE) << "OUT db get usr annotation item.";
    return 0;
}

int DB::get_all_dcm_items(std::vector<ImgItem>& items) {
    MI_IO_LOG(MI_TRACE) << "IN db get all dcm items."; 
    if (!this->is_valid()) {;
        MI_IO_LOG(MI_ERROR) << "db connection invalid.";
        return -1;
    }

    try {
        std::stringstream ss;
        ss << "SELECT * FROM " << DCM_TABLE;
        sql::PreparedStatement* pstmt =
            _connection->prepareStatement(ss.str().c_str());
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
                item.dcm_path = res->getString("dcm_path");
                item.preprocess_mask_path = res->getString("preprocess_mask_path");
                item.annotation_ai_path = res->getString("annotation_ai_path");
                item.size_mb = res->getInt("size_mb");
                item.ai_intermediate_data_path = res->getInt("ai_intermediate_data_path");
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
        MI_IO_LOG(MI_ERROR) << "db get all items failed with exception: "
        << this->get_sql_exception_info_i(&e);
        return -1;
    }
    MI_IO_LOG(MI_TRACE) << "OUT db get all dcm items."; 
    return 0;
}

int DB::get_all_usr_annotation_items(std::vector<AnnoItem>& items) {
    MI_IO_LOG(MI_TRACE) << "IN db get all usr annotation items.";
    if (!this->is_valid()) {
        MI_IO_LOG(MI_ERROR) << "db connection invalid.";
        return -1;
    }

    try {
        std::stringstream ss;
        ss << "SELECT * FROM " << ANNO_TABLE;
        sql::PreparedStatement* pstmt = _connection->prepareStatement(ss.str().c_str());
        sql::ResultSet* res = pstmt->executeQuery();

        items.clear();
        while(true) {
            if(res->next()) {
                AnnoItem item;
                item.series_id = res->getString("series_id");
                item.usr_name = res->getString("usr_name");
                item.annotation_usr_path = res->getString("annotation_usr_path");
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
        MI_IO_LOG(MI_ERROR) << "db get all usr annotation items failed with exception: "
        << this->get_sql_exception_info_i(&e);
        return -1;
    }
    MI_IO_LOG(MI_TRACE) << "OUT db get all usr annotation items.";
    return 0;
}

int DB::update_preprocess_mask(const std::string& series_id, const std::string& preprocess_mask_path) {
    MI_IO_LOG(MI_TRACE) << "IN db update preprocess mask.";
    if (!this->is_valid()) {
        MI_IO_LOG(MI_ERROR) << "db connection invalid.";
        return -1;
    }

    try {
        std::stringstream ss;
        ss << "UPDATE " << DCM_TABLE << " SET preprocess_mask_path=\'" << preprocess_mask_path << "\'" 
        << " WHERE series_id=\'" << series_id << "\'";
        sql::PreparedStatement* pstmt = _connection->prepareStatement(ss.str().c_str());
        sql::ResultSet* res = pstmt->executeQuery();
        delete res;
        res = nullptr;
        delete pstmt;
        pstmt = nullptr;
    } catch (const sql::SQLException& e) {
        MI_IO_LOG(MI_ERROR) << "db update preprocess mask failed with exception: "
        << this->get_sql_exception_info_i(&e);
        return -1;
    }
    MI_IO_LOG(MI_TRACE) << "OUT db update preprocess mask.";
    return 0;
}

int DB::update_ai_annotation(const std::string& series_id, const std::string& annotation_ai_path) {
    MI_IO_LOG(MI_TRACE) << "IN db update AI annotation.";
    if (!this->is_valid()) {
        MI_IO_LOG(MI_ERROR) << "db connection invalid.";
        return -1;
    }

    try {
        std::stringstream ss;
        ss << "UPDATE " << DCM_TABLE << " SET annotation_ai_path=\'" << annotation_ai_path << "\'" 
        << " WHERE series_id=\'" << series_id << "\'";
        sql::PreparedStatement* pstmt = _connection->prepareStatement(ss.str().c_str());
        sql::ResultSet* res = pstmt->executeQuery();
        delete res;
        res = nullptr;
        delete pstmt;
        pstmt = nullptr;
    } catch (const sql::SQLException& e) {
        MI_IO_LOG(MI_ERROR) << "db update AI annotation failed with exception: "
        << this->get_sql_exception_info_i(&e);
        return -1;
    }
    MI_IO_LOG(MI_TRACE) << "OUT db update AI annotation.";
    return 0;
}

int DB::update_ai_intermediate_data(const std::string& series_id, const std::string& ai_intermediate_data_path) {
    MI_IO_LOG(MI_TRACE) << "IN db update AI intermediate data.";
    if (!this->is_valid()) {
        MI_IO_LOG(MI_ERROR) << "db connection invalid.";
        return -1;
    }

    try {
        std::stringstream ss;
        ss << "UPDATE " << DCM_TABLE << " SET ai_intermediate_data_path=\'" << ai_intermediate_data_path << "\'" 
        << " WHERE series_id=\'" << series_id << "\'";
        sql::PreparedStatement* pstmt = _connection->prepareStatement(ss.str().c_str());
        sql::ResultSet* res = pstmt->executeQuery();
        delete res;
        res = nullptr;
        delete pstmt;
        pstmt = nullptr;
    } catch (const sql::SQLException& e) {
        MI_IO_LOG(MI_ERROR) << "db update AI intermediate data failed with exception: "
        << this->get_sql_exception_info_i(&e);
        return -1;
    }
    MI_IO_LOG(MI_TRACE) << "OUT db update AI intermediate data.";
    return 0;
}

int DB::update_usr_annotation(const std::string& series_id, const std::string& usr_name, const std::string& annotation_usr_path) {
    MI_IO_LOG(MI_TRACE) << "IN db update usr annotation.";
    if (!this->is_valid()) {
        MI_IO_LOG(MI_ERROR) << "db connection invalid.";
        return -1;
    }

    try {
        std::stringstream ss;
        ss << "UPDATE " << ANNO_TABLE << " SET annotation_usr_path=\'" << annotation_usr_path << "\'" 
        << " WHERE series_id=\'" << series_id << "\'" << " AND usr_name=\'" << usr_name;
        sql::PreparedStatement* pstmt = _connection->prepareStatement(ss.str().c_str());
        sql::ResultSet* res = pstmt->executeQuery();
        delete res;
        res = nullptr;
        delete pstmt;
        pstmt = nullptr;
    } catch (const sql::SQLException& e) {
        MI_IO_LOG(MI_ERROR) << "db update usr annotation failed with exception: "
        << this->get_sql_exception_info_i(&e);
        return -1;
    }
    MI_IO_LOG(MI_TRACE) << "OUT db update usr annotation.";
    return 0;
}

MED_IMG_END_NAMESPACE