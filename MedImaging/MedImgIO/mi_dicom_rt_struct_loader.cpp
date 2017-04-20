#include "mi_dicom_rt_struct_loader.h"
#include "mi_image_data.h"
#include "mi_image_data_header.h"
#include "mi_dicom_rt_struct.h"

#include "boost/algorithm/string.hpp"  


#include "dcmtk/config/osconfig.h"  
#include "dcmtk/dcmdata/dctk.h" 
#include "dcmtk/dcmdata/dcdicdir.h"
#include "dcmtk/dcmdata/dcdatset.h"
#include "dcmtk/dcmimgle/dcmimage.h"
#include "dcmtk/dcmdata/dctk.h" 
#include "dcmtk/dcmdata/dcpxitem.h" 
#include "dcmtk/dcmjpeg/djdecode.h"

MED_IMAGING_BEGIN_NAMESPACE

DICOMRTLoader::DICOMRTLoader()
{

}

DICOMRTLoader::~DICOMRTLoader()
{

}

IOStatus DICOMRTLoader::load_rt_struct(const std::string& sFile , std::shared_ptr<RTStruct> &pRTStruct)
{
    try
    {
        DcmFileFormatPtr pFileformat(new DcmFileFormat());
        OFCondition status = pFileformat->loadFile(sFile.c_str());
        if (status.bad())
        {
            //TODO LOG
            std::cout << "err\n";
        }

        return load_rt_struct(pFileformat , pRTStruct);

    }
    catch (const Exception& e)
    {
        std::cout << e.what();
        throw e;
    }
    
}

IOStatus DICOMRTLoader::load_rt_struct(DcmFileFormatPtr pFileformat , std::shared_ptr<RTStruct> &pRTStruct)
{
    try
    {
        pRTStruct.reset( new RTStruct());

        DcmDataset* pDataSet = pFileformat->getDataset();

        //1 Get modality
        OFString sModality;
        OFCondition status = pDataSet->findAndGetOFString(DCM_Modality , sModality);
        if (status.bad())
        {
            //TODO LOG
            std::cout << "err\n";
        }

        if ( "RTSTRUCT" != sModality)
        {
            //TODO LOG
            std::cout << "err\n";
        }
        std::cout << sModality.c_str() << std::endl;

        DcmSequenceOfItems* pROISequence = nullptr;
        status = pDataSet->findAndGetSequence(DCM_StructureSetROISequence , pROISequence);
        if (status.bad())
        {
            //TODO LOG
            std::cout << "err\n";
        }

        DcmSequenceOfItems* pContourSequence = nullptr;
        status = pDataSet->findAndGetSequence(DCM_ROIContourSequence , pContourSequence);
        if (status.bad())
        {
            //TODO LOG
            std::cout << "err\n";
        }

        const unsigned long uiROINum = pROISequence->card();
        const unsigned long uiContourNum = pContourSequence->card();
        if (uiROINum != uiContourNum)
        {
            //TODO LOG
            std::cout << "err\n";
        }

        for (unsigned long i = 0 ; i<uiROINum ; ++i)
        {
            DcmItem* pROIItem = pROISequence->getItem(i);
            DcmItem* pCoutourItem = pContourSequence->getItem(i);

            OFString sROIName;
            status = pROIItem->findAndGetOFString(DCM_ROIName ,sROIName);
            if (status.bad())
            {
                //TODO LOG
                std::cout << "err\n";
            }

            DcmSequenceOfItems* pContourUnitSequence = nullptr;
            status = pCoutourItem->findAndGetSequence(DCM_ContourSequence , pContourUnitSequence);
            if (status.bad())
            {
                //TODO LOG
                std::cout << "err\n";
            }

            unsigned long uiContourUnitNum = pContourUnitSequence->card();
            for (unsigned long j = 0 ; j<uiContourUnitNum ; ++j)
            {
                DcmItem* pCoutourUnit = pContourUnitSequence->getItem(j);
                OFString sPoints;
                status = pCoutourUnit->findAndGetOFStringArray(DCM_ContourData ,  sPoints );
                if (status.bad())
                {
                    //TODO LOG
                    std::cout << "err\n";
                }

                std::vector<std::string> vecPoints;
                boost::split(vecPoints , sPoints, boost::is_any_of("|/\\"));
                if (0 != vecPoints.size()%3)
                {
                    //TODO LOG
                }

                ContourData* pData = new ContourData();
                pData->m_vecPoints.resize(vecPoints.size()/3);
                for (int k= 0  ; k < vecPoints.size()/3 ; ++k)
                {
                    pData->m_vecPoints[k]._m[0] = (float)atof(vecPoints[k*3].c_str());
                    pData->m_vecPoints[k]._m[1] = (float)atof(vecPoints[k*3+1].c_str());
                    pData->m_vecPoints[k]._m[2] = (float)atof(vecPoints[k*3+2].c_str());
                }

                pRTStruct->add_contour(sROIName.c_str() , pData);

            }


            std::cout << sROIName << std::endl;


        }

        std::cout << "DONE\n";

        return IO_SUCCESS;

    }
    catch (const Exception& e)
    {
        pRTStruct.reset();
        std::cout << e.what();
        return IO_SUCCESS;
    }
}

MED_IMAGING_END_NAMESPACE