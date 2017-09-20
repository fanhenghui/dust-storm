#include "mi_app_data_base.h"

#include "cppconn/driver.h"
#include "cppconn/exception.h"
#include "cppconn/prepared_statement.h"
#include "cppconn/resultset.h"
#include "cppconn/sqlstring.h"
#include "cppconn/statement.h"
#include "mysql_connection.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

const static std::string IMG_TBL = "img_tbl";

AppDataBase::AppDataBase(): _connection(nullptr) {}

AppDataBase::~AppDataBase() {
    this->disconnect();
}

int AppDataBase::connect(const std::string& user, const std::string& ip_port,
                         const std::string& pwd, const std::string& db_name) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN db connect."; 
    if (user.empty()) {
        MI_APPCOMMON_LOG(MI_ERROR) << "connect DB: user is empty.";
        return -1;
    }

    if (ip_port.empty()) {
        MI_APPCOMMON_LOG(MI_ERROR) << "connect DB: ip/port is empty.";
        return -1;
    }

    if (pwd.empty()) {
        MI_APPCOMMON_LOG(MI_ERROR) << "connect DB: password is empty.";
        return -1;
    }

    if (db_name.empty()) {
        MI_APPCOMMON_LOG(MI_ERROR) << "connect DB: db name is empty.";
        return -1;
    }

    _connection = nullptr;
    try {
        // create connect
        sql::Driver* driver = get_driver_instance();
        _connection = driver->connect(ip_port.c_str(), user.c_str(), pwd.c_str());
        // _connection = driver->connect("tcp://127.0.0.1:3306", "root", "0123456");
        _connection->setSchema(db_name.c_str());
    } catch (const sql::SQLException& e) {
        std::stringstream ss;
        ss << "# ERR: " << e.what()
        << " (MySQL error code: " << e.getErrorCode()
        << ", SQLState: " << e.getSQLState() << " )";
        MI_APPCOMMON_LOG(MI_ERROR) << "connect db failed with exception: " << ss.str();
        delete _connection;
        _connection = nullptr;
        return -1;
    }

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT db connect.";
    MI_APPCOMMON_LOG(MI_INFO) << "connect to db success.";
    return 0;
}

int AppDataBase::disconnect() {
    if (_connection) {
        delete _connection;
    }
    _connection = nullptr;
    MI_APPCOMMON_LOG(MI_INFO) << "disconnect db success.";
    return 0;
}

int AppDataBase::get_series_path(const std::string& series_id, std::string& path, const std::string& tbl) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN db query get series path."; 
    if (nullptr == _connection) {
        MI_APPCOMMON_LOG(MI_ERROR) << "db connection is null.";
        return -1;
    }

    try {
        sql::Statement* stmt = _connection->createStatement();
        delete stmt;

        std::stringstream ss;
        ss << "SELECT * FROM " << tbl << " WHERE series_id=\'" << series_id
           << "\';";
        sql::PreparedStatement* pstmt =
            _connection->prepareStatement(ss.str().c_str());
        sql::ResultSet* res = pstmt->executeQuery();

        if (res->next()) {
            path = res->getString("path");
            delete res;
            res = nullptr;
            delete pstmt;
            pstmt = nullptr;
            return 0;
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
        MI_APPCOMMON_LOG(MI_ERROR) << "qurey db when get series path failed with exception: " << ss.str();
        return -1;
    }

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT db query get series path."; 
}

int AppDataBase::get_all_item(std::vector<ImgItem>& items , const std::string& tbl) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN db query get all item."; 
    if (nullptr == _connection) {
        MI_APPCOMMON_LOG(MI_ERROR) << "db connection is null.";
        return -1;
    }

    try {
        sql::Statement* stmt = _connection->createStatement();
        delete stmt;

        std::stringstream ss;
        ss << "SELECT * FROM " << tbl << ";";
        sql::PreparedStatement* pstmt =
            _connection->prepareStatement(ss.str().c_str());
        sql::ResultSet* res = pstmt->executeQuery();

        items.clear();
        std::cout << res->rowsCount() << std::endl;

        for (;;) {
            if (res->next()) {
                ImgItem item;
                item.series_id = res->getString("series_id");
                item.study_id = res->getString("study_id");
                item.patient_name = res->getString("patient_name");
                item.patient_id = res->getString("patient_id");
                item.modality = res->getString("modality");
                item.path = res->getString("path");
                items.push_back(item);
            } else {
                break;
            }
        }

        delete res;
        res = nullptr;
        delete pstmt;
        pstmt = nullptr;
        return 0;
    } catch (const sql::SQLException& e) {
        std::stringstream ss;
        ss << "# ERR: " << e.what()
        << " (MySQL error code: " << e.getErrorCode()
        << ", SQLState: " << e.getSQLState() << " )";
        MI_APPCOMMON_LOG(MI_ERROR) << "qurey db when get all items failed with exception: " << ss.str();
        return -1;
    }
    MI_APPCOMMON_LOG(MI_TRACE) << "OUT db query get all item."; 
}

int AppDataBase::insert_item(const ImgItem& item) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN db query inset item."; 
    if (nullptr == _connection) {
        MI_APPCOMMON_LOG(MI_ERROR) << "db connection is null.";
        return -1;
    }

    // write to DB (find ; delete if exit ; insert)
    try {
        sql::Statement* stmt = _connection->createStatement();
        delete stmt;
        sql::PreparedStatement* pstmt = nullptr;
        sql::ResultSet* res = nullptr;

        // find
        std::string sql_str;
        {
            std::stringstream ss;
            ss << "SELECT * FROM img_tbl where series_id=\'" << item.series_id << "\';";
            sql_str = ss.str();
        }
        pstmt = _connection->prepareStatement(sql_str.c_str());
        res = pstmt->executeQuery();
        delete pstmt;
        pstmt = nullptr;

        // delete if exit
        if (res->next()) {
            std::stringstream ss;
            ss << "WARNING : already has the same series item : " << item.series_id
               << " , use the new one replace it.\n";
            delete res;
            res = nullptr;

            // delete old one
            {
                std::stringstream ss;
                ss << "DELETE FROM img_tbl where series_id=\'" << item.series_id << "\';";
                sql_str = ss.str();
            }
            sql::PreparedStatement* pstmt = _connection->prepareStatement(sql_str.c_str());
            sql::ResultSet* res = pstmt->executeQuery();
            delete pstmt;
            pstmt = nullptr;
            delete res;
            res = nullptr;
        }

        // insert new item
        {
            std::stringstream ss;
            ss << "INSERT INTO img_tbl (series_id , study_id , patient_name , "
               "patient_id ,modality , path) values (";
            ss << "\'" << item.series_id << "\',";
            ss << "\'" << item.study_id << "\',";
            ss << "\'" << item.patient_name << "\',";
            ss << "\'" << item.patient_id << "\',";
            ss << "\'" << item.modality << "\',";
            ss << "\'" << item.path << "\'";
            ss << ");";
            sql_str = ss.str();
        }
        pstmt = _connection->prepareStatement(sql_str.c_str());
        res = pstmt->executeQuery();
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
        // TODO recovery DB

        return -1;
    }
    MI_APPCOMMON_LOG(MI_TRACE) << "OUT db query inset item."; 
}

MED_IMG_END_NAMESPACE