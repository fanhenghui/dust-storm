#include "mi_dicom_loader.h"
#include "mi_meta_object_loader.h"
#include "mi_image_data.h"
#include "mi_image_data_header.h"

using namespace medical_imaging;

namespace
{

}

void IOUT_LoadMetaObject()
{
    //std::string file_name = "E:/Data/Ali/test_subset00/test_subset00/LKDS-00012.mhd";
    std::string file_name = "E:\\Data\\Kaggle\\Luna\\subject0\\subject0\\1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd";

    MetaObjectLoader loader;
    std::shared_ptr<MetaObjectTag> meta_obj_tag;
    std::shared_ptr<ImageData> img;
    std::shared_ptr<ImageDataHeader> data_header;
    IOStatus status = loader.load(file_name , img , meta_obj_tag , data_header);
    if (status == IO_SUCCESS)
    {
        std::cout << "OK\n";
    }
    else
    {
        std::cout << "ERROR\n";
    }
}

