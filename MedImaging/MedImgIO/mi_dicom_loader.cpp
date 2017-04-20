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

IOStatus DICOMLoader::load_series(const std::vector<std::string>& vecFiles , std::shared_ptr<ImageData> &pImgData , std::shared_ptr<ImageDataHeader> &pImgDataHeader)
{
    if (vecFiles.empty())
    {
        return IO_EMPTY_INPUT;
    }

    const unsigned int uiSliceCount = static_cast<unsigned int >(vecFiles.size());
    DcmFileFormatSet vecDataFormats;

    //////////////////////////////////////////////////////////////////////////
    //1 load series
    for (auto it = vecFiles.begin() ; it != vecFiles.end() ; ++it)
    {
        const std::string sFileName = *it;
        DcmFileFormatPtr pFileformat(new DcmFileFormat());
        OFCondition status = pFileformat->loadFile(sFileName.c_str());
        if (status.bad())
        {
            return IO_FILE_OPEN_FAILED;
        }
        vecDataFormats.push_back(pFileformat);
    }

    //////////////////////////////////////////////////////////////////////////
    //2 Data check
    IOStatus eCheckingStatus = data_check_i(vecDataFormats);
    if(IO_SUCCESS !=  eCheckingStatus)
    {
        return eCheckingStatus;
    }

    if (uiSliceCount < 16)//不支持少于16张的数据进行三维可视化 // TODO 这一步在这里做不太合适
    {
        return IO_UNSUPPORTED_YET;
    }

    //////////////////////////////////////////////////////////////////////////
    //3 Sort series
    sort_series_i(vecDataFormats);

    //////////////////////////////////////////////////////////////////////////
    //4 Construct image data header
    pImgDataHeader.reset(new ImageDataHeader());
    IOStatus eDataHeadingStatus = construct_data_header_i(vecDataFormats , pImgDataHeader);
    if(IO_SUCCESS !=  eDataHeadingStatus)
    {
        pImgDataHeader.reset();
        return eDataHeadingStatus;
    }

    //////////////////////////////////////////////////////////////////////////
    //5 Construct image data
    pImgData.reset(new ImageData());
    IOStatus eDataImagingStatus = construct_image_data_i(vecDataFormats , pImgDataHeader , pImgData);
    if(IO_SUCCESS !=  eDataImagingStatus)
    {
        pImgDataHeader.reset();
        pImgData.reset();
        return eDataImagingStatus;
    }

    return IO_SUCCESS;
}

IOStatus DICOMLoader::data_check_i(DcmFileFormatSet& vecDatasets)
{
    return IO_SUCCESS;
}

void DICOMLoader::sort_series_i(DcmFileFormatSet& vecFileFormat)
{

}

IOStatus DICOMLoader::construct_data_header_i(DcmFileFormatSet& vecFileFormat , std::shared_ptr<ImageDataHeader> pImgDataHeader)
{
    IOStatus eLoadingStatus = IO_DATA_DAMAGE;
    try
    {
        DcmFileFormatPtr pFileFirst = vecFileFormat[0];
        DcmDataset* pImgFirst = pFileFirst->getDataset();
        if (!pImgFirst)
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Get data set failed!");
        }

        DcmMetaInfo* pMetaInfoFirst = pFileFirst->getMetaInfo();
        if (!pMetaInfoFirst)
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Get meta infp failed!");
        }

        //4.1 Get Transfer Syntax UID
        if (!get_transfer_syntax_uid_i(pMetaInfoFirst , pImgDataHeader))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag TransferSyntaxUID failed!");
        }

        //4.2 Get Study UID
        if (!get_study_uid_i(pImgFirst , pImgDataHeader))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag StudyUID failed!");
        }

        //4.3 Get Series UID
        if (!get_series_uid_i(pImgFirst , pImgDataHeader))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag SeriesUID failed!");
        }

        //4.4 Get Date
        if (!get_content_time_i(pImgFirst , pImgDataHeader))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag ContentTime failed!");
        }

        //4.5 Get Modality
        OFString sModality;
        OFCondition status = pImgFirst->findAndGetOFString(DCM_Modality , sModality);
        if (status.bad())
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag Modality failed!");
        }
        if ("CT" == sModality)
        {
            pImgDataHeader->m_eModality = CT;
        }
        else if ("MR" == sModality)
        {
            pImgDataHeader->m_eModality = MR;
        }
        else if ("PT" == sModality)
        {
            pImgDataHeader->m_eModality = PT;
        }
        else if ("CR" == sModality)
        {
            pImgDataHeader->m_eModality = CR;
        }
        else
        {
            eLoadingStatus = IO_UNSUPPORTED_YET;
            IO_THROW_EXCEPTION("Unsupport modality YET!");
        }

        //4.6 Manufacturer
        if (!get_manufacturer_i(pImgFirst , pImgDataHeader))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag Manufacturer failed!");
        }

        //4.7 Manufacturer model
        if (!get_manufacturer_model_name_i(pImgFirst , pImgDataHeader))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag ManufacturerModelName failed!");
        }

        //4.8 Patient name
        if (!get_patient_name_i(pImgFirst , pImgDataHeader))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag PatientName failed!");
        }

        //4.9 Patient ID
        if (!get_patient_id_i(pImgFirst , pImgDataHeader))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag PatientID failed!");
        }

        //4.10 Patient Sex(很多图像都没有这个Tag)
        if (!get_patient_sex_i(pImgFirst , pImgDataHeader))
        {
            //eLoadingStatus = IO_DATA_DAMAGE;
            //IO_THROW_EXCEPTION("Parse tag PatientSex failed!");
        }

        //4.11 Patient Age(很多图像都没有这个Tag)
        if (!get_patient_age_i(pImgFirst , pImgDataHeader))
        {
            /*eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag PatientAge failed!");*/
        }

        //4.12 Slice thickness (不是一定必要)
        if (!get_slice_thickness_i(pImgFirst , pImgDataHeader))
        {
            //eLoadingStatus = IO_DATA_DAMAGE;
            //IO_THROW_EXCEPTION("Parse tag SliceThickness failed!");
        }

        //4.13 KVP (CT only)
        if (!get_kvp_i(pImgFirst , pImgDataHeader))
        {
            //eLoadingStatus = IO_DATA_DAMAGE;
            //IO_THROW_EXCEPTION("Parse tag KVP failed!");
        }

        //4.14 Patient position
        if (!get_patient_position_i(pImgFirst , pImgDataHeader))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag PatientPositio failed!");
        }

        //4.15 Samples per Pixel
        if (!get_sample_per_pixel_i(pImgFirst , pImgDataHeader))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag SamplePerPixel failed!");
        }

        //4.16 Photometric Interpretation 
        OFString sPhotometricInterpretation;
        status = pImgFirst->findAndGetOFString(DCM_PhotometricInterpretation , sPhotometricInterpretation);
        if (status.bad())
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag PhotometricInterpretation failed!");
        }

        const std::string sPI = std::string(sPhotometricInterpretation.c_str());
        if ("MONOCHROME1" == sPI)
        {
            pImgDataHeader->m_ePhotometricInterpretation = PI_MONOCHROME1;
        }
        else if ("MONOCHROME2" == sPI)
        {
            pImgDataHeader->m_ePhotometricInterpretation = PI_MONOCHROME2;
        }
        else if ("RGB" == sPI)
        {
            pImgDataHeader->m_ePhotometricInterpretation = PI_RGB;
        }
        else
        {
            eLoadingStatus = IO_UNSUPPORTED_YET;
            IO_THROW_EXCEPTION("Unsupport photometric Interpretation YET!");
        }

        //4.17 Rows
        if (!get_rows_i(pImgFirst , pImgDataHeader))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag Rows failed!");
        }

        //4.18 Columns
        if (!get_columns_i(pImgFirst , pImgDataHeader))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag Columns failed!");
        }

        //4.19 Pixel Spacing
        if (!get_pixel_spacing_i(pImgFirst , pImgDataHeader))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag PixelSpacing failed!");
        }

        //4.20 Bits Allocated
        if (!get_bits_allocated_i(pImgFirst , pImgDataHeader))
        {
            eLoadingStatus = IO_DATA_DAMAGE;
            IO_THROW_EXCEPTION("Parse tag BitsAllocated failed!");
        }

        //4.21 Pixel Representation
        if (!get_pixel_representation_i(pImgFirst , pImgDataHeader))
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

IOStatus DICOMLoader::construct_image_data_i(DcmFileFormatSet& vecFileFormat , std::shared_ptr<ImageDataHeader> pDataHeader , std::shared_ptr<ImageData> pImgData)
{
    const unsigned int uiSliceCount= static_cast<unsigned int>(vecFileFormat.size());
    DcmFileFormatPtr pFileFirst = vecFileFormat[0];
    DcmDataset* pImgFirst = pFileFirst->getDataset();
    DcmFileFormatPtr pFileLast = vecFileFormat[uiSliceCount-1];
    DcmDataset* pImgLast = pFileLast->getDataset();

    //Intercept and slope
    get_intercept_i(pImgFirst , pImgData->m_fIntercept);
    get_slope_i(pImgFirst , pImgData->m_fSlope);

    pDataHeader->m_SliceLocations.resize(uiSliceCount);
    pDataHeader->m_ImgPositions.resize(uiSliceCount);
    for (unsigned int i = 0 ; i<uiSliceCount ; ++i)
    {
        double dSliceLoc = 0;
        Point3 ptImgPos;
        DcmDataset* dataset = vecFileFormat[i]->getDataset();
        if (!dataset)
        {
            return IO_DATA_DAMAGE;
        }

        get_slice_location_i(dataset , dSliceLoc);
        pDataHeader->m_SliceLocations[i] = dSliceLoc;

        get_image_position_i(dataset , ptImgPos);
        pDataHeader->m_ImgPositions[i] = ptImgPos;
    }

    //Data channel
    if (PI_RGB == pDataHeader->m_ePhotometricInterpretation && 3 == pDataHeader->m_uiSamplePerPixel)
    {
        pImgData->m_uiChannelNum = 3;
    }
    else if( (PI_MONOCHROME1 == pDataHeader->m_ePhotometricInterpretation ||
        PI_MONOCHROME2 == pDataHeader->m_ePhotometricInterpretation) && 1 == pDataHeader->m_uiSamplePerPixel)
    {
        pImgData->m_uiChannelNum = 1;
    }
    else
    {
        return IO_UNSUPPORTED_YET;
    }

    //Data type
    unsigned int uiImgSize = pDataHeader->m_uiImgRows*pDataHeader->m_uiImgColumns;
    if (8 == pDataHeader->m_uiBitsAllocated)
    {
        if (0 == pDataHeader->m_uiPixelRepresentation)
        {
            pImgData->m_eDataType = UCHAR;
        }
        else
        {
            pImgData->m_eDataType = CHAR;
        }
    }
    else if (16 == pDataHeader->m_uiBitsAllocated)
    {
        uiImgSize *= 2;

        if (0 == pDataHeader->m_uiPixelRepresentation)
        {
            pImgData->m_eDataType = USHORT;
        }
        else
        {
            pImgData->m_eDataType = SHORT;
        }
    }
    else
    {
        return IO_UNSUPPORTED_YET;
    }

    //Dimension
    pImgData->m_uiDim[0] = pDataHeader->m_uiImgColumns;
    pImgData->m_uiDim[1] = pDataHeader->m_uiImgRows;
    pImgData->m_uiDim[2] = uiSliceCount;

    //Spacing
    pImgData->m_dSpacing[0] = pDataHeader->m_dPixelSpacing[1];
    pImgData->m_dSpacing[1] = pDataHeader->m_dPixelSpacing[0];
    const double dSliceLocFirst= pDataHeader->m_SliceLocations[0];
    const double dSliceLocLast= pDataHeader->m_SliceLocations[uiSliceCount-1];
    pImgData->m_dSpacing[2] = abs( (dSliceLocLast - dSliceLocFirst)/static_cast<double>(uiSliceCount-1));

    //Image position in patient
    pImgData->m_ptImgPositon = pDataHeader->m_ImgPositions[0];

    //Image Orientation in patient
    Vector3 vOriRow;
    Vector3 vOriColumn;
    if(!get_image_orientation_i(pImgFirst , vOriRow , vOriColumn))
    {
        return IO_DATA_DAMAGE;
    }
    pImgData->m_vImgOrientation[0] = vOriRow;
    pImgData->m_vImgOrientation[1] = vOriColumn;
    pImgData->m_vImgOrientation[2] = pDataHeader->m_ImgPositions[uiSliceCount-1] - pDataHeader->m_ImgPositions[0];
    pImgData->m_vImgOrientation[2].normalize();

    //Image data
    pImgData->mem_allocate();
    char* pData = (char*)(pImgData->get_pixel_pointer());
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

    const std::string& ksMyTSU = pDataHeader->m_sTransferSyntaxUID;

    if (ksMyTSU == ksTSU_LittleEndianImplicitTransferSyntax ||
        ksMyTSU == ksTSU_LittleEndianExplicitTransferSyntax ||
        ksMyTSU == ksTSU_DeflatedExplicitVRLittleEndianTransferSyntax ||
        ksMyTSU == ksTSU_BigEndianExplicitTransferSyntax)
    {
        for (unsigned int i = 0 ; i<uiSliceCount ; ++i)
        {
            DcmDataset* dataset = vecFileFormat[i]->getDataset();
            if (!dataset)
            {
                return IO_DATA_DAMAGE;
            }
            get_pixel_data_i(vecFileFormat[i] , dataset , pData + uiImgSize*i , uiImgSize);
        }
    }
    else if (ksMyTSU == ksTSU_JPEGProcess14SV1TransferSyntax ||
        ksMyTSU == ksTSU_JPEGProcess14TransferSyntax)
    {
        for (unsigned int i = 0 ; i<uiSliceCount ; ++i)
        {
            DcmDataset* dataset = vecFileFormat[i]->getDataset();
            if (!dataset)
            {
                return IO_DATA_DAMAGE;
            }
            get_jpeg_compressed_pixel_data_i(vecFileFormat[i] , dataset , pData + uiImgSize*i , uiImgSize);
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



bool DICOMLoader::get_transfer_syntax_uid_i(DcmMetaInfo* pMetaInfo, std::shared_ptr<ImageDataHeader> & pImgDataHeader)
{
    OFString sTransferSynaxUID;
    OFCondition status = pMetaInfo->findAndGetOFString(DCM_TransferSyntaxUID , sTransferSynaxUID);
    if (status.bad())
    {
        return false;
    }
    pImgDataHeader->m_sTransferSyntaxUID = std::string(sTransferSynaxUID.c_str());
    return true;
}

bool DICOMLoader::get_content_time_i(DcmDataset*pImg , std::shared_ptr<ImageDataHeader> & pImgDataHeader)
{
    OFString sDate;
    OFCondition status = pImg->findAndGetOFString(DCM_ContentDate , sDate);
    if (status.bad())
    {
        return false;
    }
    pImgDataHeader->m_sImageDate = std::string(sDate.c_str());
    return true;
}

bool DICOMLoader::get_manufacturer_i(DcmDataset*pImg , std::shared_ptr<ImageDataHeader> & pImgDataHeader)
{
    OFString sManufacturer;
    OFCondition status = pImg->findAndGetOFString(DCM_Manufacturer , sManufacturer);
    if (status.bad())
    {
        return false;
    }
    pImgDataHeader->m_sManufacturer = std::string(sManufacturer.c_str());
    return true;
}

bool DICOMLoader::get_manufacturer_model_name_i(DcmDataset*pImg , std::shared_ptr<ImageDataHeader> & pImgDataHeader)
{
    OFString sManufacturerModel;
    OFCondition status = pImg->findAndGetOFString(DCM_ManufacturerModelName , sManufacturerModel);
    if (status.bad())
    {
         return false;
    }
    pImgDataHeader->m_sManufacturerModelName = std::string(sManufacturerModel.c_str());
    return true;
}

bool DICOMLoader::get_patient_name_i(DcmDataset*pImg , std::shared_ptr<ImageDataHeader> & pImgDataHeader)
{
    OFString sString;
    OFCondition status = pImg->findAndGetOFString(DCM_PatientName , sString);
    if (status.bad())
    {
        return false;
    }
    pImgDataHeader->m_sPatientName = std::string(sString.c_str());
    return true;
}

bool DICOMLoader::get_patient_id_i(DcmDataset*pImg , std::shared_ptr<ImageDataHeader> & pImgDataHeader)
{
    OFString sString;
    OFCondition status = pImg->findAndGetOFString(DCM_PatientID , sString);
    if (status.bad())
    {
        return false;
    }
    pImgDataHeader->m_sPatientID = std::string(sString.c_str());
    return true;
}

bool DICOMLoader::get_patient_sex_i(DcmDataset*pImg , std::shared_ptr<ImageDataHeader> & pImgDataHeader)
{
    OFString sString;
    OFCondition status = pImg->findAndGetOFString(DCM_PatientSex , sString);
    if (status.bad())
    {
        return false;
    }
    pImgDataHeader->m_sPatientSex = std::string(sString.c_str());
    return true;
}

bool DICOMLoader::get_patient_age_i(DcmDataset*pImg , std::shared_ptr<ImageDataHeader> & pImgDataHeader)
{
    OFString sString;
    OFCondition status = pImg->findAndGetOFString(DCM_PatientAge, sString);
    if (status.bad())
    {
        return false;
    }
    pImgDataHeader->m_sPatientAge = std::string(sString.c_str());
    return true;
}

bool DICOMLoader::get_slice_thickness_i(DcmDataset*pImg , std::shared_ptr<ImageDataHeader> & pImgDataHeader)
{
    OFString sString;
    OFCondition status = pImg->findAndGetOFString(DCM_SliceThickness, sString);
    if (status.bad())
    {
        return false;
    }
    pImgDataHeader->m_dSliceThickness = static_cast<double>(atof(sString.c_str()));
    return true;
}

bool DICOMLoader::get_kvp_i(DcmDataset*pImg , std::shared_ptr<ImageDataHeader> & pImgDataHeader)
{
    OFString sString;
    OFCondition status = pImg->findAndGetOFString(DCM_KVP, sString);
    if (status.bad())
    {
        return false;
    }
    pImgDataHeader->m_kVp = static_cast<float>(atof(sString.c_str()));
    return true;
}

bool DICOMLoader::get_patient_position_i(DcmDataset*pImg , std::shared_ptr<ImageDataHeader> & pImgDataHeader)
{
    OFString sString;
    OFCondition status = pImg->findAndGetOFString(DCM_PatientPosition, sString);
    if (status.bad())
    {
        return false;
    }
    const std::string sPP = std::string(sString.c_str());

    if ("HFP" == sPP)
    {
        pImgDataHeader->m_ePatientPosition = HFP;
    }
    else if("HFS" == sPP)
    {
        pImgDataHeader->m_ePatientPosition = HFS;
    }
    else if("HFDR" == sPP)
    {
        pImgDataHeader->m_ePatientPosition = HFDR;
    }
    else if("HFDL" == sPP)
    {
        pImgDataHeader->m_ePatientPosition = HFDL;
    }
    else if ("FFP" == sPP)
    {
        pImgDataHeader->m_ePatientPosition = FFP;
    }
    else if("FFS" == sPP)
    {
        pImgDataHeader->m_ePatientPosition = FFS;
    }
    else if("FFDR" == sPP)
    {
        pImgDataHeader->m_ePatientPosition = FFDR;
    }
    else if("FFDL" == sPP)
    {
        pImgDataHeader->m_ePatientPosition = FFDL;
    }

    return true;
}

bool DICOMLoader::get_series_uid_i(DcmDataset*pImg , std::shared_ptr<ImageDataHeader> & pImgDataHeader)
{
    OFString sSeriesUID;
    OFCondition status = pImg->findAndGetOFString(DCM_SeriesInstanceUID , sSeriesUID);
    if (status.bad())
    {
        return false;
    }
    pImgDataHeader->m_sSeriesUID = std::string(sSeriesUID.c_str());
    return true;
}

bool DICOMLoader::get_study_uid_i(DcmDataset*pImg , std::shared_ptr<ImageDataHeader> & pImgDataHeader)
{
    OFString sStudyUID;
    OFCondition status = pImg->findAndGetOFString(DCM_StudyInstanceUID , sStudyUID);
    if (status.bad())
    {
        return false;
    }
    pImgDataHeader->m_sStudyUID = std::string(sStudyUID.c_str());
    return true;
}

bool DICOMLoader::get_sample_per_pixel_i(DcmDataset*pImg , std::shared_ptr<ImageDataHeader> & pImgDataHeader)
{
    unsigned short usValue = 0;
    OFCondition status = pImg->findAndGetUint16(DCM_SamplesPerPixel , usValue);
    if (status.bad())
    {
        return false;
    }
    pImgDataHeader->m_uiSamplePerPixel = static_cast<unsigned int>(usValue);
    return true;
}

bool DICOMLoader::get_rows_i(DcmDataset*pImg , std::shared_ptr<ImageDataHeader> & pImgDataHeader)
{
    unsigned short usValue = 0;
    OFCondition status = pImg->findAndGetUint16(DCM_Rows , usValue);
    if (status.bad())
    {
        return false;
    }
    pImgDataHeader->m_uiImgRows = static_cast<unsigned int>(usValue);
    return true;
}

bool DICOMLoader::get_columns_i(DcmDataset*pImg , std::shared_ptr<ImageDataHeader> & pImgDataHeader)
{
    unsigned short usValue = 0;
    OFCondition status = pImg->findAndGetUint16(DCM_Columns , usValue);
    if (status.bad())
    {
        return false;
    }
    pImgDataHeader->m_uiImgColumns = static_cast<unsigned int>(usValue);
    return true;
}

bool DICOMLoader::get_pixel_spacing_i(DcmDataset*pImg , std::shared_ptr<ImageDataHeader> & pImgDataHeader)
{
    OFString sRowspacing, sColspacing;
    OFCondition status1 =pImg->findAndGetOFString(DCM_PixelSpacing , sRowspacing , 0);
    OFCondition status2 =pImg->findAndGetOFString(DCM_PixelSpacing , sColspacing , 1);
    if (status1.bad() || status2.bad())
    {
        return false;
    }
    pImgDataHeader->m_dPixelSpacing[0] = atof(sRowspacing.c_str());
    pImgDataHeader->m_dPixelSpacing[1] = atof(sColspacing.c_str());

    return true;
}

bool DICOMLoader::get_bits_allocated_i(DcmDataset*pImg , std::shared_ptr<ImageDataHeader> & pImgDataHeader)
{
    unsigned short usValue = 0;
    OFCondition status = pImg->findAndGetUint16(DCM_BitsAllocated , usValue);
    if (status.bad())
    {
        return false;
    }
    pImgDataHeader->m_uiBitsAllocated = static_cast<unsigned int>(usValue);
    return true;
}

bool DICOMLoader::get_pixel_representation_i(DcmDataset*pImg , std::shared_ptr<ImageDataHeader> & pImgDataHeader)
{
    unsigned short usValue = 0;
    OFCondition status = pImg->findAndGetUint16(DCM_PixelRepresentation , usValue);
    if (status.bad())
    {
        return false;
    }
    pImgDataHeader->m_uiPixelRepresentation = static_cast<unsigned int>(usValue);
    return true;
}

bool DICOMLoader::get_intercept_i(DcmDataset*pImg , float& fIntercept)
{
    OFString sString;
    OFCondition status = pImg->findAndGetOFString(DCM_RescaleIntercept , sString);
    if (status.bad())
    {
        return false;
    }
    fIntercept = (float)atof(sString.c_str());
    return true;
}

bool DICOMLoader::get_slope_i(DcmDataset*pImg , float& fSlope)
{
    OFString sString;
    OFCondition status = pImg->findAndGetOFString(DCM_RescaleSlope , sString);
    if (status.bad())
    {
        return false;
    }
    fSlope = (float)atof(sString.c_str());
    return true;
}

bool DICOMLoader::get_instance_number_i(DcmDataset*pImg , int& iInstanceNumber)
{
    OFString sString;
    OFCondition status = pImg->findAndGetOFString(DCM_InstanceNumber , sString);
    if (status.bad())
    {
        return false;
    }
    iInstanceNumber = atoi(sString.c_str());
    return true;
}

bool DICOMLoader::get_image_position_i(DcmDataset*pImg , Point3& ptImgPos)
{
    OFString sImagePos;
    OFCondition status = pImg->findAndGetOFString(DCM_ImagePositionPatient , sImagePos , 0);
    if (status.bad())
    {
        return false;
    }
    ptImgPos.x = static_cast<double>(atof(sImagePos.c_str()));

    status = pImg->findAndGetOFString(DCM_ImagePositionPatient , sImagePos , 1);
    if (status.bad())
    {
        return false;
    }
    ptImgPos.y = static_cast<double>(atof(sImagePos.c_str()));

    status = pImg->findAndGetOFString(DCM_ImagePositionPatient , sImagePos , 2);
    if (status.bad())
    {
        return false;
    }
    ptImgPos.z = static_cast<double>(atof(sImagePos.c_str()));

    return true;
}

bool DICOMLoader::get_image_orientation_i(DcmDataset*pImg , Vector3& vRow , Vector3& vColumn)
{
    double dImgOri[6] = {0};
    for (int i = 0 ; i <6 ; ++i)
    {
        OFString sImageOri;
        OFCondition status = pImg->findAndGetOFString(DCM_ImageOrientationPatient , sImageOri , i);
        if (status.bad())
        {
            return false;
        }
        dImgOri[i] = static_cast<double>(atof(sImageOri.c_str()));
    }

    vRow = Vector3(dImgOri[0],dImgOri[1],dImgOri[2]);
    vColumn = Vector3(dImgOri[3],dImgOri[4],dImgOri[5]);

    return true;
}

bool DICOMLoader::get_slice_location_i(DcmDataset*pImg , double& dSliceLoc)
{
    OFString sString;
    OFCondition status = pImg->findAndGetOFString(DCM_SliceLocation , sString);
    if (status.bad())
    {
        return false;
    }
    dSliceLoc = atoi(sString.c_str());
    return true;
}

bool DICOMLoader::get_pixel_data_i(DcmFileFormatPtr pFileFormat , DcmDataset*pImg , char* pData , unsigned int uiSize)
{
    const unsigned char* pReadData;
    OFCondition status = pImg->findAndGetUint8Array(DCM_PixelData , pReadData);
    if (status.bad())
    {
        return false;
    }
    memcpy(pData , pReadData ,uiSize);

    return true;
}

bool DICOMLoader::get_jpeg_compressed_pixel_data_i(DcmFileFormatPtr pFileFormat , DcmDataset*pImg , char* pData , unsigned int uiSize)
{
    //Code from : http://support.dcmtk.org/docs/mod_dcmjpeg.html
    //Write to a temp decompressed file , then read the decompressed one

    DJDecoderRegistration::registerCodecs(); // register JPEG codecs

    // decompress data set if compressed
    pImg->chooseRepresentation(EXS_LittleEndianExplicit, NULL);

    // check if everything went well
    if (pImg->canWriteXfer(EXS_LittleEndianExplicit))
    {
        pFileFormat->saveFile("test_decompressed.dcm", EXS_LittleEndianExplicit);
    }
    DJDecoderRegistration::cleanup(); // deregister JPEG codecs

    pFileFormat->loadFile("test_decompressed.dcm");
    DcmDataset* pDataSet = pFileFormat->getDataset();

    return get_pixel_data_i(pFileFormat , pDataSet , pData , uiSize);
}

MED_IMAGING_END_NAMESPACE