#ifndef MEDIMGIO_MI_MYSQL_DB_H
#define MEDIMGIO_MI_MYSQL_DB_H

#include <vector>
#include "io/mi_io_export.h"

namespace sql {
    class Connection;
    class SQLException;
}

MED_IMG_BEGIN_NAMESPACE

class IO_Export MySQLDB {
public:
    enum MySQLDataType {
        INT,
        UINT,
        INT64,
        UINT64,
        DOUBLE,
        STRING,
    };

    MySQLDB();
    virtual ~MySQLDB();

    int connect(const std::string& user, const std::string& ip_port,
                const std::string& pwd, const std::string& db_name);
    int disconnect();
    
    bool is_valid();

    bool reconnect();

    bool try_connect();
    
protected:
    sql::Connection* _connection;
    std::string _db_name;    

protected:
    std::string get_sql_exception_info(const sql::SQLException* exception);
    
private:
    DISALLOW_COPY_AND_ASSIGN(MySQLDB);
};

MED_IMG_END_NAMESPACE

#endif