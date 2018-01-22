#include "io/mi_dicom_rt_struct.h"
#include "io/mi_dicom_rt_struct_loader.h"
#include "log/mi_logger.h"
#include <string>

using namespace medical_imaging;

int rt_ut(int argc, char* argv[]) {
    std::string path("/home/wangrui22/data/rt/000000.dcm");
    if (argc == 2 ) {
        path = std::string(argv[1]);
    }
    
    MI_LOG(MI_INFO) << "parse RT dicom file: " << path;
    
    DICOMRTLoader rt_loader;
    std::shared_ptr<RTStruct> rt(new RTStruct());
    if( IO_SUCCESS != rt_loader.load_rt_struct(path, rt) ){
         MI_LOG(MI_ERROR) << "load RT dicom error.\n";
    } else {
        rt->write_to_file("/home/wangrui22/data/rt_result.txt");
    }

    return 0;
}