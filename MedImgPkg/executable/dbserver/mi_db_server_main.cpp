#include <stdio.h>
#include <stdlib.h>
#include <signal.h>

#include "util/mi_exception.h"
#include "io/mi_configure.h"

#include "mi_db_server_controller.h"
#include "mi_db_server_logger.h"

using namespace medical_imaging;
std::shared_ptr<DBServerController> controller(new DBServerController());

void dbs_exit() {
    //quiet logic
};

int main(int argc , char* argv[]) {
    try {
        const std::string log_config_file = Configure::instance()->get_log_config_file();
        Logger::instance()->bind_config_file(log_config_file);

        Logger::instance()->set_file_name_format("logs/mi-db-%Y-%m-%d_%H-%M-%S.%N.log");
        Logger::instance()->set_file_direction("");
        Logger::instance()->initialize();
        MI_DBSERVER_LOG(MI_INFO) << "hello DB server.";

        controller->initialize();
        //atexit(dbs_exit);
        signal(SIGINT, SIG_IGN);//ignore Ctrl+C
        controller->run();
        controller->finalize();
    } catch (Exception& e) {
        MI_DBSERVER_LOG(MI_FATAL) << "DB server error exit with exception: " << e.what();
        return -1;
    }
    catch (std::exception& e) {
        MI_DBSERVER_LOG(MI_FATAL) << "DB server error exit with exception: " << e.what();
        return -1;
    }
    catch (...) {
        MI_DBSERVER_LOG(MI_FATAL) << "DB server error exit with unknown exception.";
        return -1;
    }
    MI_DBSERVER_LOG(MI_INFO) << "bye DB server.";
    return 0;
}