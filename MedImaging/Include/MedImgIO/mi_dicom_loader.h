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

    IO_Export IOStatus load_series(const std::vector<std::string>& vecFiles , std::shared_ptr<ImageData> &pImgData , std::shared_ptr<ImageDataHeader> & pImgDataHeader);

private:
    IOStatus data_check_i(DcmFileFormatSet& vecFileFormat);

    void sort_series_i(DcmFileFormatSet& vecFileFormat);

    IOStatus construct_data_header_i(DcmFileFormatSet& vecFileFormat , std::shared_ptr<ImageDataHeader> pImgDataHeader);

    IOStatus construct_image_data_i(DcmFileFormatSet& vecFileFormat , std::shared_ptr<ImageDataHeader> pImgDataHeader , std::shared_ptr<ImageData> pImgData);

private:
    bool get_transfer_syntax_uid_i(DcmMetaInfo* pMetaInfo ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool get_content_time_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool get_manufacturer_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool get_manufacturer_model_name_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool get_patient_name_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool get_patient_id_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool get_patient_sex_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool get_patient_age_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool get_slice_thickness_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool get_kvp_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool get_patient_position_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool get_series_uid_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool get_study_uid_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool get_sample_per_pixel_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool get_rows_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool get_columns_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool get_pixel_spacing_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool get_bits_allocated_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool get_pixel_representation_i(DcmDataset*pImg ,  std::shared_ptr<ImageDataHeader> & pImgDataHeader);

    bool get_intercept_i(DcmDataset*pImg ,  float& fIntercept);

    bool get_slope_i(DcmDataset*pImg ,  float& fSlope);

    bool get_instance_number_i(DcmDataset*pImg , int& iInstanceNumber);

    bool get_image_position_i(DcmDataset*pImg , Point3& ptImgPos);

    bool get_image_orientation_i(DcmDataset*pImg , Vector3& vRow , Vector3& vColumn);

    bool get_slice_location_i(DcmDataset*pImg , double& dSliceLoc);

    bool get_pixel_data_i(DcmFileFormatPtr pFileFormat , DcmDataset*pImg , char* pData , unsigned int uiSize);

    bool get_jpeg_compressed_pixel_data_i(DcmFileFormatPtr pFileFormat , DcmDataset*pImg , char* pData , unsigned int uiSize);
};

MED_IMAGING_END_NAMESPACE
#endif