#include <stdio.h>
#include <stdlib.h>

#include "util/mi_exception.h"

#include "mi_db_server_controller.h"
#include "mi_db_server_logger.h"
#include "appcommon/mi_app_config.h"

using namespace medical_imaging;

int main(int argc , char* argv[]) {
    try {
        const std::string log_config_file = AppConfig::instance()->get_log_config_file();
        Logger::instance()->bind_config_file(log_config_file);

        Logger::instance()->set_file_name_format("logs/mi-db-%Y-%m-%d_%H-%M-%S.%N.log");
        Logger::instance()->set_file_direction("");
        Logger::instance()->initialize();
        MI_DBSERVER_LOG(MI_INFO) << "hello db server.";

        std::shared_ptr<DBServerController> controller(new DBServerController());
        controller->initialize();
        controller->run();
        controller->finalize();
    } catch (Exception& e) {
        MI_DBSERVER_LOG(MI_FATAL) << "db server error exit with exception: " << e.what();
        return -1;
    }
    catch (std::exception& e) {
        MI_DBSERVER_LOG(MI_FATAL) << "db server error exit with exception: " << e.what();
        return -1;
    }
    catch (...) {
        MI_DBSERVER_LOG(MI_FATAL) << "db server error exit with unknown exception.";
        return -1;
    }
    MI_DBSERVER_LOG(MI_INFO) << "bye db server.";
    return 0;
}