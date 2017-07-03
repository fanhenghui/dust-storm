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

MED_IMG_BEGIN_NAMESPACE

DICOMRTLoader::DICOMRTLoader()
{

}

DICOMRTLoader::~DICOMRTLoader()
{

}

IOStatus DICOMRTLoader::load_rt_struct(const std::string& file_name , std::shared_ptr<RTStruct> &rt_struct)
{
    try
    {
        DcmFileFormatPtr file_format(new DcmFileFormat());
        OFCondition status = file_format->loadFile(file_name.c_str());
        if (status.bad())
        {
            //TODO LOG
            std::cout << "err\n";
        }

        return load_rt_struct(file_format , rt_struct);

    }
    catch (const Exception& e)
    {
        std::cout << e.what();
        throw e;
    }
    
}

IOStatus DICOMRTLoader::load_rt_struct(DcmFileFormatPtr file_format , std::shared_ptr<RTStruct> &rt_struct)
{
    try
    {
        rt_struct.reset( new RTStruct());

        DcmDataset* data_set = file_format->getDataset();

        //1 Get modality
        OFString modality;
        OFCondition status = data_set->findAndGetOFString(DCM_Modality , modality);
        if (status.bad())
        {
            //TODO LOG
            std::cout << "err\n";
        }

        if ( "RTSTRUCT" != modality)
        {
            //TODO LOG
            std::cout << "err\n";
        }
        std::cout << modality.c_str() << std::endl;

        DcmSequenceOfItems* roi_sequence = nullptr;
        status = data_set->findAndGetSequence(DCM_StructureSetROISequence , roi_sequence);
        if (status.bad())
        {
            //TODO LOG
            std::cout << "err\n";
        }

        DcmSequenceOfItems* contour_sequence = nullptr;
        status = data_set->findAndGetSequence(DCM_ROIContourSequence , contour_sequence);
        if (status.bad())
        {
            //TODO LOG
            std::cout << "err\n";
        }

        const unsigned long roi_num = roi_sequence->card();
        const unsigned long contour_num = contour_sequence->card();
        if (roi_num != contour_num)
        {
            //TODO LOG
            std::cout << "err\n";
        }

        for (unsigned long i = 0 ; i<roi_num ; ++i)
        {
            DcmItem* roi_item = roi_sequence->getItem(i);
            DcmItem* coutour_item = contour_sequence->getItem(i);

            OFString roi_name;
            status = roi_item->findAndGetOFString(DCM_ROIName ,roi_name);
            if (status.bad())
            {
                //TODO LOG
                std::cout << "err\n";
            }

            DcmSequenceOfItems* contour_unit_sequence = nullptr;
            status = coutour_item->findAndGetSequence(DCM_ContourSequence , contour_unit_sequence);
            if (status.bad())
            {
                //TODO LOG
                std::cout << "err\n";
            }

            unsigned long contour_unit_num = contour_unit_sequence->card();
            for (unsigned long j = 0 ; j<contour_unit_num ; ++j)
            {
                DcmItem* coutour_unit = contour_unit_sequence->getItem(j);
                OFString points_array;
                status = coutour_unit->findAndGetOFStringArray(DCM_ContourData ,  points_array );
                if (status.bad())
                {
                    //TODO LOG
                    std::cout << "err\n";
                }

                std::vector<std::string> points;
                boost::split(points , points_array, boost::is_any_of("|/\\"));
                if (0 != points.size()%3)
                {
                    //TODO LOG
                }

                ContourData* contour_data = new ContourData();
                contour_data->points.resize(points.size()/3);
                for (int k= 0  ; k < points.size()/3 ; ++k)
                {
                    contour_data->points[k]._m[0] = (float)atof(points[k*3].c_str());
                    contour_data->points[k]._m[1] = (float)atof(points[k*3+1].c_str());
                    contour_data->points[k]._m[2] = (float)atof(points[k*3+2].c_str());
                }

                rt_struct->add_contour(roi_name.c_str() , contour_data);

            }


            std::cout << roi_name << std::endl;


        }

        std::cout << "DONE\n";

        return IO_SUCCESS;

    }
    catch (const Exception& e)
    {
        rt_struct.reset();
        std::cout << e.what();
        return IO_SUCCESS;
    }
}

MED_IMG_END_NAMESPACE