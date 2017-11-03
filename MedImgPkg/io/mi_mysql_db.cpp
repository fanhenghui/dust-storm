#include "mi_mysql_db.h"

#include "cppconn/driver.h"
#include "cppconn/exception.h"
#include "cppconn/prepared_statement.h"
#include "cppconn/resultset.h"
#include "cppconn/sqlstring.h"
#include "cppconn/statement.h"
#include "mysql_connection.h"

#include "mi_io_logger.h"

MED_IMG_BEGIN_NAMESPACE

MySQLDB::MySQLDB():_connection(nullptr),_db_name("") {

}
MySQLDB::~MySQLDB() {
    this->disconnect();
}

int MySQLDB::connect(const std::string& user, const std::string& ip_port, const std::string& pwd, const std::string& db_name){
    MI_IO_LOG(MI_TRACE) << "IN db connect."; 
    if (user.empty()) {
        MI_IO_LOG(MI_ERROR) << "connect DB: user is empty.";
        return -1;
    }

    if (ip_port.empty()) {
        MI_IO_LOG(MI_ERROR) << "connect DB: ip/port is empty.";
        return -1;
    }

    if (pwd.empty()) {
        MI_IO_LOG(MI_ERROR) << "connect DB: password is empty.";
        return -1;
    }

    if (db_name.empty()) {
        MI_IO_LOG(MI_ERROR) << "connect DB: db name is empty.";
        return -1;
    }

    _connection = nullptr;
    _db_name = db_name;
    try {
        sql::Driver* driver = get_driver_instance();
        const std::string host_name = "tcp://" + ip_port;
        _connection = driver->connect(host_name.c_str(), user.c_str(), pwd.c_str());
        _connection->setSchema(db_name.c_str());
        if (!this->is_valid()) {
            MI_IO_LOG(MI_ERROR) << "connection to mysql is dead.";
            return -1;
        }
    } catch (const sql::SQLException& e) {
        MI_IO_LOG(MI_ERROR) << "connect db failed with exception: " 
        << this->get_sql_exception_info_i(&e);
        delete _connection;
        _connection = nullptr;
        return -1;
    }

    MI_IO_LOG(MI_TRACE) << "OUT db connect.";
    MI_IO_LOG(MI_INFO) << "connect to db success.";
    return 0;
}

int MySQLDB::disconnect() {
    if (_connection) {
        delete _connection;
    }
    _connection = nullptr;
    _db_name = "";
    MI_IO_LOG(MI_INFO) << "disconnect db success.";
    return 0;
}

bool MySQLDB::is_valid() {
    if (nullptr == _connection) {
        MI_IO_LOG(MI_ERROR) << "sql connection is null.";
        return false;
    } else {
        return _connection->isValid();
    }
}

bool MySQLDB::reconnect() {
    if (nullptr == _connection) {
        MI_IO_LOG(MI_ERROR) << "sql connection is null.";
        return false;
    } else {
        const bool flag = _connection->reconnect();
        if (flag) {
            _connection->setSchema(_db_name.c_str());
        }
        return flag;
    }
}

std::string MySQLDB::get_sql_exception_info_i(const sql::SQLException* e) {
    if (nullptr == e) {
        MI_IO_LOG(MI_ERROR) << "SQL exception is null.";
        return "";
    } else {
        std::stringstream ss;
        ss << "# ERR: " << e->what()
        << " (MySQL error code: " << e->getErrorCode()
        << ", SQLState: " << e->getSQLState() << " )";
        return ss.str();
    }
}

MED_IMG_END_NAMESPACE

