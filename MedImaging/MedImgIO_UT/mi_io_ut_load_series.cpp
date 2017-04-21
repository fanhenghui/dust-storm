#include "mi_dicom_loader.h"
#include "mi_image_data.h"
#include "mi_image_data_header.h"


using namespace medical_imaging;

namespace
{
    std::vector<std::string> GetFiles()
    {
        const std::string file_name = "D:/Data/MyData/AB_CTA_01/";
        unsigned int uiSliceCount = 734;
        const std::string sPrefix ="DICOM7_000";
        std::string sCurFile;
        std::vector<std::string> files;
        for (unsigned int i = 0 ; i< uiSliceCount ; ++i)
        {
            std::stringstream ss;
            if (i<10)
            {
                ss << file_name << sPrefix << "00" << i;
            }
            else if (i<100)
            {
                ss << file_name << sPrefix << "0" << i;
            }
            else
            {
                ss << file_name << sPrefix  << i;
            }
            files.push_back(ss.str());
        }

        return files;
    }
}

void IOUT_LoadSeries()
{
    std::shared_ptr<ImageDataHeader> data_header;
    std::shared_ptr<ImageData> image_data;

    std::vector<std::string> files = GetFiles();
    DICOMLoader loader;
    IOStatus status = loader.load_series(files , image_data , data_header);

    if (IO_SUCCESS == status)
    {
        std::ofstream out("D:/Data/MyData/AB_CTA_01.raw" , std::ios::binary | std::ios::out);
        if (out.is_open())
        {
            out.write((char*)image_data->get_pixel_pointer() , image_data->_dim[0]*image_data->_dim[1]*image_data->_dim[2]*2);
            out.close();
        }
    }

}

