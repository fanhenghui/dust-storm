#ifndef MED_IMAGING_DICOM_LOADER_H
#define MED_IMAGING_DICOM_LOADER_H

#include "MedImgIO/mi_io_stdafx.h"
#include "MedImgCommon/mi_common_define.h"
#include "MedImgArithmetic/mi_point3.h"
#include "MedImgArithmetic/mi_vector3.h"

class DcmFileFormat;
class DcmDataset;
class DcmMetaInfo;

typedef std::shared_ptr<DcmFileFormat> DcmFileFormatPtr;
typedef std::vector<DcmFileFormatPtr> DcmFileFormatSet;

MED_IMAGING_BEGIN_NAMESPACE

class ImageData;
class ImageDataHeader;

class  DICOMLoader
{
public:
    IO_Export DICOMLoader();

    IO_Export ~DICOMLoader();

    IO_Export IOStatus LoadSeries(const std::vector<std::string>& vecFiles , std::shared_ptr<ImageData> &pImgData , std::shared_ptr<ImageDataHeader> & pImgDataHeader);

private:
    IOStatus DataCheck_i(DcmFileFormatSet& vecFileFormat);

    void SortSeries_i(DcmFileFormatSet& vecFileFormat);

    IOStatus ConstructDataHeader_i(DcmFileFormatSet& vecFileFormat , std::shared_ptr<ImageDataHeader> pImgDataHeader);

    IOStatus ConstructImageData_i(DcmFileFormatSet& vecFileFormat , std::shared_ptr<ImageDataHeader> pImgDataHeader , std::shared_ptr<ImageData> pImgData);

private:
    bool GetTransferSyntaxUID_i(DcmMetaInfo* pMetaInfo ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool GetContentTime_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool GetManufacturer_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool GetManufacturerModelName_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool GetPatientName_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool GetPatientID_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool GetPatientSex_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool GetPatientAge_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool GetSliceThickness_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool GetKVP_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool GetPatientPosition_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool GetSeriesUID_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool GetStudyUID_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool GetSamplePerPixel_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool GetRows_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool GetColumns_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool GetPixelSpacing_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool GetBitsAllocated_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool GetPixelRepresentation_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool GetIntercept_i(DcmDataset*pImg ,  float& fIntercept);

    bool GetSlope_i(DcmDataset*pImg ,  float& fSlope);

    bool GetInstanceNumber_i(DcmDataset*pImg , int& iInstanceNumber);

    bool GetImagePosition_i(DcmDataset*pImg , Point3& ptImgPos);

    bool GetImageOrientation_i(DcmDataset*pImg , Vector3& vRow , Vector3& vColumn);

    bool GetSliceLocation_i(DcmDataset*pImg , double& dSliceLoc);

    bool GetPixelData_i(DcmFileFormatPtr pFileFormat , DcmDataset*pImg , char* pData , unsigned int uiSize);

    bool GetJPEGCompressedPixelData_i(DcmFileFormatPtr pFileFormat , DcmDataset*pImg , char* pData , unsigned int uiSize);
};

MED_IMAGING_END_NAMESPACE
#endif