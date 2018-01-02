#include <string>
#include "log/mi_logger.h"

extern int pacs_ut(int argc, char* argv[]);
extern int targa_ut(int argc, char* argv[]);
extern int dicom_loader_ut(int argc, char* argv[]);
extern int md5_ut(int argc, char* argv[]);

using namespace medical_imaging;

int main(int argc, char* argv[]) {

    const std::string log_config_file = "../config/log_cofig";
    Logger::instance()->bind_config_file(log_config_file);
    Logger::instance()->set_file_name_format("logs/mi-io-ut-%Y-%m-%d_%H-%M-%S.%N.log");
    Logger::instance()->set_file_direction("");
    Logger::instance()->initialize();

    return md5_ut(argc,argv);
}