#ifndef MED_IMG_DB_SERVER_ECHO_H
#define MED_IMG_DB_SERVER_ECHO_H

#include "mi_db_server_common.h"

MED_IMG_BEGIN_NAMESPACE

class DBServerController;
class ConsoleEcho;
class DBServerConsoleEcho {
public:
    explicit DBServerConsoleEcho(std::shared_ptr<DBServerController>);
    ~DBServerConsoleEcho();
    void run(); 

private:
    void init();
    std::weak_ptr<DBServerController> _controller;
    std::unique_ptr<ConsoleEcho> _console_echo;
private:
    DISALLOW_COPY_AND_ASSIGN(DBServerConsoleEcho);
};

MED_IMG_END_NAMESPACE

#endif