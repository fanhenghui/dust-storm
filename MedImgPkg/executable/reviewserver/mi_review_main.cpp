#include <stdio.h>
#include <stdlib.h>

#include "util/mi_exception.h"

#include "mi_review_controller.h"
#include "mi_review_logger.h"
#include "appcommon/mi_app_config.h"

#ifdef WIN32
#else
#include <libgen.h>
#endif

using namespace medical_imaging;

int main(int argc , char* argv[]) {
    try {
#ifndef WIN32
        chdir(dirname(argv[0]));
#endif

        const std::string log_config_file = AppConfig::instance()->get_log_config_file();
        Logger::instance()->bind_config_file(log_config_file);
        std::string path = (argc == 2) ? argv[1] : "";
        std::string usr_name = path;
        for (int i = static_cast<int>(path.size()) - 1; i >= 0; --i) {
            if (path[i] == '/') {
                usr_name = path.substr(i+1 , path.size() - i - 1);
                break;
            }
        }

        Logger::instance()->set_file_name_format("logs/mi-" + usr_name + "-%Y-%m-%d_%H-%M-%S.%N.log");
        Logger::instance()->set_file_direction("");
        Logger::instance()->initialize();
        MI_REVIEW_LOG(MI_INFO) << "hello review server.";

        MI_REVIEW_LOG(MI_INFO) << "path is " << path;
        MI_REVIEW_LOG(MI_INFO) << "usr name is " << usr_name;
        if(path.empty()) {
            MI_REVIEW_LOG(MI_FATAL) << "path is empty.";
            return -1;
        }

        std::shared_ptr<ReviewController> controller(new ReviewController());
        controller->initialize();
        controller->run(path);
        controller->finalize();
    } catch (Exception& e) {
        MI_REVIEW_LOG(MI_FATAL) << "review server error exit with exception: " << e.what();
        return -1;
    }
    catch (std::exception& e) {
        MI_REVIEW_LOG(MI_FATAL) << "review server error exit with exception: " << e.what();
        return -1;
    }
    catch (...) {
        MI_REVIEW_LOG(MI_FATAL) << "review server error exit with unknown exception.";
        return -1;
    }
    MI_REVIEW_LOG(MI_INFO) << "bye review server.";
    return 0;
}