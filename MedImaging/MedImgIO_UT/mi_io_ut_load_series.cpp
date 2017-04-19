#include "mi_dicom_loader.h"
#include "mi_image_data.h"
#include "mi_image_data_header.h"


using namespace MedImaging;

namespace
{
    std::vector<std::string> GetFiles()
    {
        const std::string sFile = "D:/Data/MyData/AB_CTA_01/";
        unsigned int uiSliceCount = 734;
        const std::string sPrefix ="DICOM7_000";
        std::string sCurFile;
        std::vector<std::string> vecFiles;
        for (unsigned int i = 0 ; i< uiSliceCount ; ++i)
        {
            std::stringstream ss;
            if (i<10)
            {
                ss << sFile << sPrefix << "00" << i;
            }
            else if (i<100)
            {
                ss << sFile << sPrefix << "0" << i;
            }
            else
            {
                ss << sFile << sPrefix  << i;
            }
            vecFiles.push_back(ss.str());
        }

        return vecFiles;
    }
}

void IOUT_LoadSeries()
{
    std::shared_ptr<ImageDataHeader> pDataHeader;
    std::shared_ptr<ImageData> pImgData;

    std::vector<std::string> vecFiles = GetFiles();
    DICOMLoader loader;
    IOStatus status = loader.LoadSeries(vecFiles , pImgData , pDataHeader);

    if (IO_SUCCESS == status)
    {
        std::ofstream out("D:/Data/MyData/AB_CTA_01.raw" , std::ios::binary | std::ios::out);
        if (out.is_open())
        {
            out.write((char*)pImgData->GetPixelPointer() , pImgData->m_uiDim[0]*pImgData->m_uiDim[1]*pImgData->m_uiDim[2]*2);
            out.close();
        }
    }

}

