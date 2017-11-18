#include <string>
#include <vector>
#include "util/mi_file_util.h"
#include "io/mi_dicom_loader.h"
#include "io/mi_image_data.h"
#include "io/mi_image_data_header.h"

using namespace medical_imaging;
int dicom_loader_ut(int argc, char* argv[]) { 
    if (argc != 2) {
        return -1;
    }

    const std::string root = argv[1];
    std::set<std::string> postfix;
    postfix.insert(".dcm");
    std::vector<std::string> files;
    FileUtil::get_all_file_recursion(root,postfix,files);

    std::vector<DICOMLoader::DCMSliceStream*> buffers;
    for (size_t i=0; i< files.size(); ++i) {
        char* buffer = nullptr;
        unsigned int size = 0;
        FileUtil::read_raw_ext(files[i],buffer,size);
        buffers.push_back(new DICOMLoader::DCMSliceStream(buffer,size));
    }

    DICOMLoader loader;
    std::shared_ptr<ImageData> img(new ImageData());
    std::shared_ptr<ImageDataHeader> header(new ImageDataHeader());
    if(IO_SUCCESS != loader.load_series(buffers,img,header) ){
        printf("load failed");
    }

    FileUtil::write_raw("/home/wangrui22/data/test.raw", (char*)img->get_pixel_pointer(), img->get_data_size());

    return 0;
}