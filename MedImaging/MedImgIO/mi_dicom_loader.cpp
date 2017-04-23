#include "mi_dicom_loader.h"
#include "mi_image_data.h"
#include "mi_image_data_header.h"

#include "dcmtk/config/osconfig.h"  
#include "dcmtk/dcmdata/dctk.h" 
#include "dcmtk/dcmdata/dcdicdir.h"
#include "dcmtk/dcmdata/dcdatset.h"
#include "dcmtk/dcmimgle/dcmimage.h"
#include "dcmtk/dcmdata/dctk.h" 
#include "dcmtk/dcmdata/dcpxitem.h" 
#include "dcmtk/dcmjpeg/djdecode.h"

MED_IMAGING_BEGIN_NAMESPACE

DICOMLoader::DICOMLoader()
{

}

DICOMLoader::~DICOMLoader()
{

}

IOStatus DICOMLoader::load_series(const std::vector<std::string>& files , std::shared_ptr<ImageData> &image_data , std::shared_ptr<ImageDataHeader> &img_data_header)
{
    clock_t start_time = clock();
    if (files.empty())
    {
        return IO_EMPTY_INPUT;
    }

    const unsigned int uiSliceCount = static_cast<unsigned int >(files.size());
    DcmFileFormatSet data_format_set;

    //////////////////////////////////////////////////////////////////////////
    //1 load series
    for (auto it = files.begin() ; it != files.end() ; ++it)
    {
        const std::string file_name = *it;
        DcmFileFormatPtr file_format(new DcmFileFormat());
        OFCondition status = file_format->loadFile(file_name.c_str());
        if (status.bad())
        {
            return IO_FILE_OPEN_FAILED;
        }
        data_format_set.push_back(file_format);
    }

    //////////////////////////////////////////////////////////////////////////
    //2 Data check
    IOStatus checking_status = data_check_i(data_format_set);
    if(IO_SUCCESS !=  checking_status)
    {
        return checking_status;
    }

    if (uiSliceCount < 16)//不支持少于16张的数据进行三维可视化 // TODO 这一步在这里做不太合适
    {
        return IO_UNSUPPORTED_YET;
    }

    //////////////////////////////////////////////////////////////////////////
    //3 Sort series
    sort_series_i(data_format_set);

    //////////////////////////////////////////////////////////////////////////
    //4 Construct image data header
    img_data_header.reset(new ImageDataHeader());
    IOStatus data_heading_status = construct_data_header_i(data_format_set , img_data_header);
    if(IO_SUCCESS !=  data_heading_status)
    {
        img_data_header.reset();
        return data_heading_status;
    }

    //////////////////////////////////////////////////////////////////////////
    //5 Construct image data
    image_data.reset(new ImageData());
    IOStatus data_imaging_status = construct_image_data_i(data_format_set , img_data_header , image_data);
    if(IO_SUCCESS !=  data_imaging_status)
    {
        img_data_header.reset();
        image_data.reset();
        return data_imaging_status;
    }

    clock_t end_time = clock();

    std::cout << "Load DICOM cost : " << double(end_time - start_time) << " ms\n";
    return IO_SUCCESS;
}

IOStatus DICOMLoader::data_check_i(DcmFileFormatSet& file_format_set)
{
    return IO_SUCCESS;
}

void DICOMLoader::sort_series_i(DcmFileFormatSet& file_format_set)
{

}

IOStatus DICOMLoader::construct_data_header_i(DcmFileFormatSet& file_format_set , std::shared_ptr<ImageDataHeader> img_data_header)
{
    IOStatus eLoadingStatus = IO_DATA_DAMAGE;
    try
    {
        DcmFileFormatPtr file_format_first = file_format_set[0];
        DcmDataset* data_set_first = file_format_first->getDataset();
        if (!data_set_first)
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Get data set failed!");
        }

        DcmMetaInfo* meta_info_first = file_format_first->getMetaInfo();
        if (!meta_info_first)
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Get meta infp failed!");
        }

        //4.1 Get Transfer Syntax UID
        if (!get_transfer_syntax_uid_i(meta_info_first , img_data_header))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag TransferSyntaxUID failed!");
        }

        //4.2 Get Study UID
        if (!get_study_uid_i(data_set_first , img_data_header))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag StudyUID failed!");
        }

        //4.3 Get Series UID
        if (!get_series_uid_i(data_set_first , img_data_header))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag SeriesUID failed!");
        }

        //4.4 Get Date
        if (!get_content_time_i(data_set_first , img_data_header))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag ContentTime failed!");
        }

        //4.5 Get Modality
        OFString modality;
        OFCondition status = data_set_first->findAndGetOFString(DCM_Modality , modality);
        if (status.bad())
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag Modality failed!");
        }
        if ("CT" == modality)
        {
            img_data_header->modality = CT;
        }
        else if ("MR" == modality)
        {
            img_data_header->modality = MR;
        }
        else if ("PT" == modality)
        {
            img_data_header->modality = PT;
        }
        else if ("CR" == modality)
        {
            img_data_header->modality = CR;
        }
        else
        {
            eLoadingStatus = IO_UNSUPPORTED_YET;
            IO_THROW_EXCEPTION("Unsupport modality YET!");
        }

        //4.6 Manufacturer
        if (!get_manufacturer_i(data_set_first , img_data_header))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag Manufacturer failed!");
        }

        //4.7 Manufacturer model
        if (!get_manufacturer_model_name_i(data_set_first , img_data_header))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag ManufacturerModelName failed!");
        }

        //4.8 Patient name
        if (!get_patient_name_i(data_set_first , img_data_header))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag PatientName failed!");
        }

        //4.9 Patient ID
        if (!get_patient_id_i(data_set_first , img_data_header))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag PatientID failed!");
        }

        //4.10 Patient Sex(很多图像都没有这个Tag)
        if (!get_patient_sex_i(data_set_first , img_data_header))
        {
            //eLoadingStatus = IO_DATA_DAMAGE;
            //IO_THROW_EXCEPTION("Parse tag PatientSex failed!");
        }

        //4.11 Patient Age(很多图像都没有这个Tag)
        if (!get_patient_age_i(data_set_first , img_data_header))
        {
            /*eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag PatientAge failed!");*/
        }

        //4.12 Slice thickness (不是一定必要)
        if (!get_slice_thickness_i(data_set_first , img_data_header))
        {
            //eLoadingStatus = IO_DATA_DAMAGE;
            //IO_THROW_EXCEPTION("Parse tag SliceThickness failed!");
        }

        //4.13 KVP (CT only)
        if (!get_kvp_i(data_set_first , img_data_header))
        {
            //eLoadingStatus = IO_DATA_DAMAGE;
            //IO_THROW_EXCEPTION("Parse tag KVP failed!");
        }

        //4.14 Patient position
        if (!get_patient_position_i(data_set_first , img_data_header))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag PatientPositio failed!");
        }

        //4.15 Samples per Pixel
        if (!get_sample_per_pixel_i(data_set_first , img_data_header))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag SamplePerPixel failed!");
        }

        //4.16 Photometric Interpretation 
        OFString sPhotometricInterpretation;
        status = data_set_first->findAndGetOFString(DCM_PhotometricInterpretation , sPhotometricInterpretation);
        if (status.bad())
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag PhotometricInterpretation failed!");
        }

        const std::string sPI = std::string(sPhotometricInterpretation.c_str());
        if ("MONOCHROME1" == sPI)
        {
            img_data_header->photometric_interpretation = PI_MONOCHROME1;
        }
        else if ("MONOCHROME2" == sPI)
        {
            img_data_header->photometric_interpretation = PI_MONOCHROME2;
        }
        else if ("RGB" == sPI)
        {
            img_data_header->photometric_interpretation = PI_RGB;
        }
        else
        {
            eLoadingStatus = IO_UNSUPPORTED_YET;
            IO_THROW_EXCEPTION("Unsupport photometric Interpretation YET!");
        }

        //4.17 Rows
        if (!get_rows_i(data_set_first , img_data_header))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag Rows failed!");
        }

        //4.18 Columns
        if (!get_columns_i(data_set_first , img_data_header))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag Columns failed!");
        }

        //4.19 Pixel Spacing
        if (!get_pixel_spacing_i(data_set_first , img_data_header))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag PixelSpacing failed!");
        }

        //4.20 Bits Allocated
        if (!get_bits_allocated_i(data_set_first , img_data_header))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag BitsAllocated failed!");
        }

        //4.21 Pixel Representation
        if (!get_pixel_representation_i(data_set_first , img_data_header))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag PixelRepresentation failed!");
        }

        return IO_SUCCESS;
    }
    catch (const Exception& e)
    {
        //TODO LOG
        std::cout << e.what();
        return eLoadingStatus;
    }
    catch(const std::exception& e)
    {
        //TODO LOG
        std::cout << e.what();
        return eLoadingStatus;
    }
}

IOStatus DICOMLoader::construct_image_data_i(DcmFileFormatSet& file_format_set , std::shared_ptr<ImageDataHeader> data_header , std::shared_ptr<ImageData> image_data)
{
    const unsigned int uiSliceCount= static_cast<unsigned int>(file_format_set.size());
    DcmFileFormatPtr file_format_first = file_format_set[0];
    DcmDataset* data_set_first = file_format_first->getDataset();
    DcmFileFormatPtr pFileLast = file_format_set[uiSliceCount-1];
    DcmDataset* pImgLast = pFileLast->getDataset();

    //Intercept and slope
    get_intercept_i(data_set_first , image_data->_intercept);
    get_slope_i(data_set_first , image_data->_slope);

    data_header->slice_location.resize(uiSliceCount);
    data_header->image_position.resize(uiSliceCount);
    for (unsigned int i = 0 ; i<uiSliceCount ; ++i)
    {
        double slice_location = 0;
        Point3 image_position;
        DcmDataset* dataset = file_format_set[i]->getDataset();
        if (!dataset)
        {
            return IO_DATA_DAMAGE;
        }

        get_slice_location_i(dataset , slice_location);
        data_header->slice_location[i] = slice_location;

        get_image_position_i(dataset , image_position);
        data_header->image_position[i] = image_position;
    }

    //Data channel
    if (PI_RGB == data_header->photometric_interpretation && 3 == data_header->sample_per_pixel)
    {
        image_data->_channel_num = 3;
    }
    else if( (PI_MONOCHROME1 == data_header->photometric_interpretation ||
        PI_MONOCHROME2 == data_header->photometric_interpretation) && 1 == data_header->sample_per_pixel)
    {
        image_data->_channel_num = 1;
    }
    else
    {
        return IO_UNSUPPORTED_YET;
    }

    //Data type
    unsigned int uiImgSize = data_header->rows*data_header->columns;
    if (8 == data_header->bits_allocated)
    {
        if (0 == data_header->pixel_representation)
        {
            image_data->_data_type = UCHAR;
        }
        else
        {
            image_data->_data_type = CHAR;
        }
    }
    else if (16 == data_header->bits_allocated)
    {
        uiImgSize *= 2;

        if (0 == data_header->pixel_representation)
        {
            image_data->_data_type = USHORT;
        }
        else
        {
            image_data->_data_type = SHORT;
        }
    }
    else
    {
        return IO_UNSUPPORTED_YET;
    }

    //Dimension
    image_data->_dim[0] = data_header->columns;
    image_data->_dim[1] = data_header->rows;
    image_data->_dim[2] = uiSliceCount;

    //Spacing
    image_data->_spacing[0] = data_header->pixel_spacing[1];
    image_data->_spacing[1] = data_header->pixel_spacing[0];
    const double dSliceLocFirst= data_header->slice_location[0];
    const double dSliceLocLast= data_header->slice_location[uiSliceCount-1];
    image_data->_spacing[2] = abs( (dSliceLocLast - dSliceLocFirst)/static_cast<double>(uiSliceCount-1));

    //Image position in patient
    image_data->_image_position = data_header->image_position[0];

    //Image Orientation in patient
    Vector3 vOriRow;
    Vector3 vOriColumn;
    if(!get_image_orientation_i(data_set_first , vOriRow , vOriColumn))
    {
        return IO_DATA_DAMAGE;
    }
    image_data->_image_orientation[0] = vOriRow;
    image_data->_image_orientation[1] = vOriColumn;
    image_data->_image_orientation[2] = data_header->image_position[uiSliceCount-1] - data_header->image_position[0];
    image_data->_image_orientation[2].normalize();

    //Image data
    image_data->mem_allocate();
    char* data_array = (char*)(image_data->get_pixel_pointer());
    //DICOM transfer syntaxes
    const std::string ksTSU_LittleEndianImplicitTransferSyntax     = std::string("1.2.840.10008.1.2");//Default transfer for DICOM
    const std::string ksTSU_LittleEndianExplicitTransferSyntax    = std::string("1.2.840.10008.1.2.1");
    const std::string ksTSU_DeflatedExplicitVRLittleEndianTransferSyntax = std::string("1.2.840.10008.1.2.1.99");
    const std::string ksTSU_BigEndianExplicitTransferSyntax = std::string("1.2.840.10008.1.2.2");

    //JEPG Lossless
    const std::string ksTSU_JPEGProcess14SV1TransferSyntax      = std::string("1.2.840.10008.1.2.4.70");//Default Transfer Syntax for Lossless JPEG Image Compression
    const std::string ksTSU_JPEGProcess14TransferSyntax     = std::string("1.2.840.10008.1.2.4.57");

    //JEPG2000 需要购买商业版的 dcmtk
    const std::string ksTSU_JEPG2000CompressionLosslessOnly = std::string("1.2.840.10008.1.2.4.90");
    const std::string ksTSU_JEPG2000Compression = std::string("1.2.840.10008.1.2.4.91");

    const std::string& ksMyTSU = data_header->transfer_syntax_uid;

    if (ksMyTSU == ksTSU_LittleEndianImplicitTransferSyntax ||
        ksMyTSU == ksTSU_LittleEndianExplicitTransferSyntax ||
        ksMyTSU == ksTSU_DeflatedExplicitVRLittleEndianTransferSyntax ||
        ksMyTSU == ksTSU_BigEndianExplicitTransferSyntax)
    {
        for (unsigned int i = 0 ; i<uiSliceCount ; ++i)
        {
            DcmDataset* dataset = file_format_set[i]->getDataset();
            if (!dataset)
            {
                return IO_DATA_DAMAGE;
            }
            get_pixel_data_i(file_format_set[i] , dataset , data_array + uiImgSize*i , uiImgSize);
        }
    }
    else if (ksMyTSU == ksTSU_JPEGProcess14SV1TransferSyntax ||
        ksMyTSU == ksTSU_JPEGProcess14TransferSyntax)
    {
        for (unsigned int i = 0 ; i<uiSliceCount ; ++i)
        {
            DcmDataset* dataset = file_format_set[i]->getDataset();
            if (!dataset)
            {
                return IO_DATA_DAMAGE;
            }
            get_jpeg_compressed_pixel_data_i(file_format_set[i] , dataset , data_array + uiImgSize*i , uiImgSize);
        }
    }
    else if (ksMyTSU == ksTSU_JEPG2000CompressionLosslessOnly ||
        ksTSU_JEPG2000Compression == ksTSU_JEPG2000Compression)
    {
        return IO_UNSUPPORTED_YET;
    }
    else
    {
        return IO_UNSUPPORTED_YET;
    }

    return IO_SUCCESS;
}



bool DICOMLoader::get_transfer_syntax_uid_i(DcmMetaInfo* meta_info, std::shared_ptr<ImageDataHeader> & img_data_header)
{
    OFString sTransferSynaxUID;
    OFCondition status = meta_info->findAndGetOFString(DCM_TransferSyntaxUID , sTransferSynaxUID);
    if (status.bad())
    {
        return false;
    }
    img_data_header->transfer_syntax_uid = std::string(sTransferSynaxUID.c_str());
    return true;
}

bool DICOMLoader::get_content_time_i(DcmDataset*data_set , std::shared_ptr<ImageDataHeader> & img_data_header)
{
    OFString sDate;
    OFCondition status = data_set->findAndGetOFString(DCM_ContentDate , sDate);
    if (status.bad())
    {
        return false;
    }
    img_data_header->image_date = std::string(sDate.c_str());
    return true;
}

bool DICOMLoader::get_manufacturer_i(DcmDataset*data_set , std::shared_ptr<ImageDataHeader> & img_data_header)
{
    OFString sManufacturer;
    OFCondition status = data_set->findAndGetOFString(DCM_Manufacturer , sManufacturer);
    if (status.bad())
    {
        return false;
    }
    img_data_header->manufacturer = std::string(sManufacturer.c_str());
    return true;
}

bool DICOMLoader::get_manufacturer_model_name_i(DcmDataset*data_set , std::shared_ptr<ImageDataHeader> & img_data_header)
{
    OFString sManufacturerModel;
    OFCondition status = data_set->findAndGetOFString(DCM_ManufacturerModelName , sManufacturerModel);
    if (status.bad())
    {
         return false;
    }
    img_data_header->manufacturer_model_name = std::string(sManufacturerModel.c_str());
    return true;
}

bool DICOMLoader::get_patient_name_i(DcmDataset*data_set , std::shared_ptr<ImageDataHeader> & img_data_header)
{
    OFString sString;
    OFCondition status = data_set->findAndGetOFString(DCM_PatientName , sString);
    if (status.bad())
    {
        return false;
    }
    img_data_header->patient_name = std::string(sString.c_str());
    return true;
}

bool DICOMLoader::get_patient_id_i(DcmDataset*data_set , std::shared_ptr<ImageDataHeader> & img_data_header)
{
    OFString sString;
    OFCondition status = data_set->findAndGetOFString(DCM_PatientID , sString);
    if (status.bad())
    {
        return false;
    }
    img_data_header->patient_id = std::string(sString.c_str());
    return true;
}

bool DICOMLoader::get_patient_sex_i(DcmDataset*data_set , std::shared_ptr<ImageDataHeader> & img_data_header)
{
    OFString sString;
    OFCondition status = data_set->findAndGetOFString(DCM_PatientSex , sString);
    if (status.bad())
    {
        return false;
    }
    img_data_header->patient_sex = std::string(sString.c_str());
    return true;
}

bool DICOMLoader::get_patient_age_i(DcmDataset*data_set , std::shared_ptr<ImageDataHeader> & img_data_header)
{
    OFString sString;
    OFCondition status = data_set->findAndGetOFString(DCM_PatientAge, sString);
    if (status.bad())
    {
        return false;
    }
    img_data_header->patient_age = std::string(sString.c_str());
    return true;
}

bool DICOMLoader::get_slice_thickness_i(DcmDataset*data_set , std::shared_ptr<ImageDataHeader> & img_data_header)
{
    OFString sString;
    OFCondition status = data_set->findAndGetOFString(DCM_SliceThickness, sString);
    if (status.bad())
    {
        return false;
    }
    img_data_header->slice_thickness = static_cast<double>(atof(sString.c_str()));
    return true;
}

bool DICOMLoader::get_kvp_i(DcmDataset*data_set , std::shared_ptr<ImageDataHeader> & img_data_header)
{
    OFString sString;
    OFCondition status = data_set->findAndGetOFString(DCM_KVP, sString);
    if (status.bad())
    {
        return false;
    }
    img_data_header->kvp = static_cast<float>(atof(sString.c_str()));
    return true;
}

bool DICOMLoader::get_patient_position_i(DcmDataset*data_set , std::shared_ptr<ImageDataHeader> & img_data_header)
{
    OFString sString;
    OFCondition status = data_set->findAndGetOFString(DCM_PatientPosition, sString);
    if (status.bad())
    {
        return false;
    }
    const std::string sPP = std::string(sString.c_str());

    if ("HFP" == sPP)
    {
        img_data_header->patient_position = HFP;
    }
    else if("HFS" == sPP)
    {
        img_data_header->patient_position = HFS;
    }
    else if("HFDR" == sPP)
    {
        img_data_header->patient_position = HFDR;
    }
    else if("HFDL" == sPP)
    {
        img_data_header->patient_position = HFDL;
    }
    else if ("FFP" == sPP)
    {
        img_data_header->patient_position = FFP;
    }
    else if("FFS" == sPP)
    {
        img_data_header->patient_position = FFS;
    }
    else if("FFDR" == sPP)
    {
        img_data_header->patient_position = FFDR;
    }
    else if("FFDL" == sPP)
    {
        img_data_header->patient_position = FFDL;
    }

    return true;
}

bool DICOMLoader::get_series_uid_i(DcmDataset*data_set , std::shared_ptr<ImageDataHeader> & img_data_header)
{
    OFString sSeriesUID;
    OFCondition status = data_set->findAndGetOFString(DCM_SeriesInstanceUID , sSeriesUID);
    if (status.bad())
    {
        return false;
    }
    img_data_header->series_uid = std::string(sSeriesUID.c_str());
    return true;
}

bool DICOMLoader::get_study_uid_i(DcmDataset*data_set , std::shared_ptr<ImageDataHeader> & img_data_header)
{
    OFString sStudyUID;
    OFCondition status = data_set->findAndGetOFString(DCM_StudyInstanceUID , sStudyUID);
    if (status.bad())
    {
        return false;
    }
    img_data_header->study_uid = std::string(sStudyUID.c_str());
    return true;
}

bool DICOMLoader::get_sample_per_pixel_i(DcmDataset*data_set , std::shared_ptr<ImageDataHeader> & img_data_header)
{
    unsigned short usValue = 0;
    OFCondition status = data_set->findAndGetUint16(DCM_SamplesPerPixel , usValue);
    if (status.bad())
    {
        return false;
    }
    img_data_header->sample_per_pixel = static_cast<unsigned int>(usValue);
    return true;
}

bool DICOMLoader::get_rows_i(DcmDataset*data_set , std::shared_ptr<ImageDataHeader> & img_data_header)
{
    unsigned short usValue = 0;
    OFCondition status = data_set->findAndGetUint16(DCM_Rows , usValue);
    if (status.bad())
    {
        return false;
    }
    img_data_header->rows = static_cast<unsigned int>(usValue);
    return true;
}

bool DICOMLoader::get_columns_i(DcmDataset*data_set , std::shared_ptr<ImageDataHeader> & img_data_header)
{
    unsigned short usValue = 0;
    OFCondition status = data_set->findAndGetUint16(DCM_Columns , usValue);
    if (status.bad())
    {
        return false;
    }
    img_data_header->columns = static_cast<unsigned int>(usValue);
    return true;
}

bool DICOMLoader::get_pixel_spacing_i(DcmDataset*data_set , std::shared_ptr<ImageDataHeader> & img_data_header)
{
    OFString sRowspacing, sColspacing;
    OFCondition status1 =data_set->findAndGetOFString(DCM_PixelSpacing , sRowspacing , 0);
    OFCondition status2 =data_set->findAndGetOFString(DCM_PixelSpacing , sColspacing , 1);
    if (status1.bad() || status2.bad())
    {
        return false;
    }
    img_data_header->pixel_spacing[0] = atof(sRowspacing.c_str());
    img_data_header->pixel_spacing[1] = atof(sColspacing.c_str());

    return true;
}

bool DICOMLoader::get_bits_allocated_i(DcmDataset*data_set , std::shared_ptr<ImageDataHeader> & img_data_header)
{
    unsigned short usValue = 0;
    OFCondition status = data_set->findAndGetUint16(DCM_BitsAllocated , usValue);
    if (status.bad())
    {
        return false;
    }
    img_data_header->bits_allocated = static_cast<unsigned int>(usValue);
    return true;
}

bool DICOMLoader::get_pixel_representation_i(DcmDataset*data_set , std::shared_ptr<ImageDataHeader> & img_data_header)
{
    unsigned short usValue = 0;
    OFCondition status = data_set->findAndGetUint16(DCM_PixelRepresentation , usValue);
    if (status.bad())
    {
        return false;
    }
    img_data_header->pixel_representation = static_cast<unsigned int>(usValue);
    return true;
}

bool DICOMLoader::get_intercept_i(DcmDataset*data_set , float& intercept)
{
    OFString sString;
    OFCondition status = data_set->findAndGetOFString(DCM_RescaleIntercept , sString);
    if (status.bad())
    {
        return false;
    }
    intercept = (float)atof(sString.c_str());
    return true;
}

bool DICOMLoader::get_slope_i(DcmDataset*data_set , float& slope)
{
    OFString sString;
    OFCondition status = data_set->findAndGetOFString(DCM_RescaleSlope , sString);
    if (status.bad())
    {
        return false;
    }
    slope = (float)atof(sString.c_str());
    return true;
}

bool DICOMLoader::get_instance_number_i(DcmDataset*data_set , int& instance_num)
{
    OFString sString;
    OFCondition status = data_set->findAndGetOFString(DCM_InstanceNumber , sString);
    if (status.bad())
    {
        return false;
    }
    instance_num = atoi(sString.c_str());
    return true;
}

bool DICOMLoader::get_image_position_i(DcmDataset*data_set , Point3& image_position)
{
    OFString sImagePos;
    OFCondition status = data_set->findAndGetOFString(DCM_ImagePositionPatient , sImagePos , 0);
    if (status.bad())
    {
        return false;
    }
    image_position.x = static_cast<double>(atof(sImagePos.c_str()));

    status = data_set->findAndGetOFString(DCM_ImagePositionPatient , sImagePos , 1);
    if (status.bad())
    {
        return false;
    }
    image_position.y = static_cast<double>(atof(sImagePos.c_str()));

    status = data_set->findAndGetOFString(DCM_ImagePositionPatient , sImagePos , 2);
    if (status.bad())
    {
        return false;
    }
    image_position.z = static_cast<double>(atof(sImagePos.c_str()));

    return true;
}

bool DICOMLoader::get_image_orientation_i(DcmDataset*data_set , Vector3& row_orientation , Vector3& column_orientation)
{
    double dImgOri[6] = {0};
    for (int i = 0 ; i <6 ; ++i)
    {
        OFString sImageOri;
        OFCondition status = data_set->findAndGetOFString(DCM_ImageOrientationPatient , sImageOri , i);
        if (status.bad())
        {
            return false;
        }
        dImgOri[i] = static_cast<double>(atof(sImageOri.c_str()));
    }

    row_orientation = Vector3(dImgOri[0],dImgOri[1],dImgOri[2]);
    column_orientation = Vector3(dImgOri[3],dImgOri[4],dImgOri[5]);

    return true;
}

bool DICOMLoader::get_slice_location_i(DcmDataset*data_set , double& slice_location)
{
    OFString sString;
    OFCondition status = data_set->findAndGetOFString(DCM_SliceLocation , sString);
    if (status.bad())
    {
        return false;
    }
    slice_location = atoi(sString.c_str());
    return true;
}

bool DICOMLoader::get_pixel_data_i(DcmFileFormatPtr pFileFormat , DcmDataset*data_set , char* data_array , unsigned int length)
{
    const unsigned char* pReadData;
    OFCondition status = data_set->findAndGetUint8Array(DCM_PixelData , pReadData);
    if (status.bad())
    {
        return false;
    }
    memcpy(data_array , pReadData ,length);

    return true;
}

bool DICOMLoader::get_jpeg_compressed_pixel_data_i(DcmFileFormatPtr pFileFormat , DcmDataset*data_set , char* data_array , unsigned int length)
{
    //Code from : http://support.dcmtk.org/docs/mod_dcmjpeg.html
    //Write to a temp decompressed file , then read the decompressed one

    DJDecoderRegistration::registerCodecs(); // register JPEG codecs

    // decompress data set if compressed
    data_set->chooseRepresentation(EXS_LittleEndianExplicit, NULL);

    // check if everything went well
    if (data_set->canWriteXfer(EXS_LittleEndianExplicit))
    {
        pFileFormat->saveFile("test_decompressed.dcm", EXS_LittleEndianExplicit);
    }
    DJDecoderRegistration::cleanup(); // deregister JPEG codecs

    pFileFormat->loadFile("test_decompressed.dcm");
    DcmDataset* pDataSet = pFileFormat->getDataset();

    return get_pixel_data_i(pFileFormat , pDataSet , data_array , length);
}

MED_IMAGING_END_NAMESPACE