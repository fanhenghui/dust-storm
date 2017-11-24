#include <stdio.h>
#include <stdlib.h>

#include "util/mi_exception.h"

#include "mi_ai_server_controller.h"
#include "mi_ai_server_logger.h"
#include "appcommon/mi_app_config.h"

using namespace medical_imaging;

int main(int argc , char* argv[]) {
    try {
        const std::string log_config_file = AppConfig::instance()->get_log_config_file();
        Logger::instance()->bind_config_file(log_config_file);

        Logger::instance()->set_file_name_format("logs/mi-ai-%Y-%m-%d_%H-%M-%S.%N.log");
        Logger::instance()->set_file_direction("");
        Logger::instance()->initialize();
        MI_AISERVER_LOG(MI_INFO) << "hello AI server.";

        std::shared_ptr<AIServerController> controller(new AIServerController());
        controller->initialize();
        controller->run();
        controller->finalize();
    } catch (Exception& e) {
        MI_AISERVER_LOG(MI_FATAL) << "AI server error exit with exception: " << e.what();
        return -1;
    }
    catch (std::exception& e) {
        MI_AISERVER_LOG(MI_FATAL) << "AI server error exit with exception: " << e.what();
        return -1;
    }
    catch (...) {
        MI_AISERVER_LOG(MI_FATAL) << "AI server error exit with unknown exception.";
        return -1;
    }
    MI_AISERVER_LOG(MI_INFO) << "bye AI server.";
    return 0;
}