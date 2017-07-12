#ifndef MED_IMG_DICOM_LOADER_H
#define MED_IMG_DICOM_LOADER_H

#include <memory>

#include "MedImgIO/mi_io_export.h"
#include "MedImgIO/mi_io_define.h"
#include "MedImgArithmetic/mi_point3.h"
#include "MedImgArithmetic/mi_vector3.h"

class DcmFileFormat;
class DcmDataset;
class DcmMetaInfo;

typedef std::shared_ptr<DcmFileFormat> DcmFileFormatPtr;
typedef std::vector<DcmFileFormatPtr> DcmFileFormatSet;

MED_IMG_BEGIN_NAMESPACE

class ImageData;
class ImageDataHeader;
class ProgressModel;

class  DICOMLoader
{
public:
    IO_Export DICOMLoader();

    IO_Export ~DICOMLoader();

    //files will removal invalid file and keep majority series
    IO_Export IOStatus load_series(std::vector<std::string>& files_in_out , std::shared_ptr<ImageData> &image_data , std::shared_ptr<ImageDataHeader> & img_data_header);

    IO_Export void set_progress_model(std::shared_ptr<ProgressModel> model);

private:
    IOStatus data_check_i(std::vector<std::string>& files , DcmFileFormatSet& file_format_set);

    IOStatus sort_series_i(DcmFileFormatSet& file_format_set);

    IOStatus construct_data_header_i(DcmFileFormatSet& file_format_set , std::shared_ptr<ImageDataHeader> img_data_header);

    IOStatus construct_image_data_i(DcmFileFormatSet& file_format_set , std::shared_ptr<ImageDataHeader> img_data_header , std::shared_ptr<ImageData> image_data);

private:
    bool get_transfer_syntax_uid_i(DcmMetaInfo* meta_info ,  std::shared_ptr<ImageDataHeader> & img_data_header);

    bool get_content_time_i(DcmDataset*data_set ,  std::shared_ptr<ImageDataHeader> & img_data_header);

    bool get_manufacturer_i(DcmDataset*data_set ,  std::shared_ptr<ImageDataHeader> & img_data_header);

    bool get_manufacturer_model_name_i(DcmDataset*data_set ,  std::shared_ptr<ImageDataHeader> & img_data_header);

    bool get_patient_name_i(DcmDataset*data_set ,  std::shared_ptr<ImageDataHeader> & img_data_header);

    bool get_patient_id_i(DcmDataset*data_set ,  std::shared_ptr<ImageDataHeader> & img_data_header);

    bool get_patient_sex_i(DcmDataset*data_set ,  std::shared_ptr<ImageDataHeader> & img_data_header);

    bool get_patient_age_i(DcmDataset*data_set ,  std::shared_ptr<ImageDataHeader> & img_data_header);

    bool get_slice_thickness_i(DcmDataset*data_set ,  std::shared_ptr<ImageDataHeader> & img_data_header);

    bool get_kvp_i(DcmDataset*data_set ,  std::shared_ptr<ImageDataHeader> & img_data_header);

    bool get_patient_position_i(DcmDataset*data_set ,  std::shared_ptr<ImageDataHeader> & img_data_header);

    bool get_series_uid_i(DcmDataset*data_set ,  std::shared_ptr<ImageDataHeader> & img_data_header);

    bool get_study_uid_i(DcmDataset*data_set ,  std::shared_ptr<ImageDataHeader> & img_data_header);

    bool get_sample_per_pixel_i(DcmDataset*data_set ,  std::shared_ptr<ImageDataHeader> & img_data_header);

    bool get_rows_i(DcmDataset*data_set ,  std::shared_ptr<ImageDataHeader> & img_data_header);

    bool get_columns_i(DcmDataset*data_set ,  std::shared_ptr<ImageDataHeader> & img_data_header);

    bool get_pixel_spacing_i(DcmDataset*data_set ,  std::shared_ptr<ImageDataHeader> & img_data_header);

    bool get_bits_allocated_i(DcmDataset*data_set ,  std::shared_ptr<ImageDataHeader> & img_data_header);

    bool get_pixel_representation_i(DcmDataset*data_set ,  std::shared_ptr<ImageDataHeader> & img_data_header);

    bool get_intercept_i(DcmDataset*data_set ,  float& intercept);

    bool get_slope_i(DcmDataset*data_set ,  float& slope);

    bool get_instance_number_i(DcmDataset*data_set , int& instance_num);

    bool get_image_position_i(DcmDataset*data_set , Point3& image_position);

    bool get_image_orientation_i(DcmDataset*data_set , Vector3& row_orientation , Vector3& column_orientation);

    bool get_slice_location_i(DcmDataset*data_set , double& slice_location);

    bool get_pixel_data_i(DcmFileFormatPtr pFileFormat , DcmDataset*data_set , char* data_array , unsigned int length);

    bool get_jpeg_compressed_pixel_data_i(DcmFileFormatPtr pFileFormat , DcmDataset*data_set , char* data_array , unsigned int length);

private:
    void add_progress_i(float value);
    void set_progress_i(int value);

    float _progress;
    std::shared_ptr<ProgressModel> _model;
};

MED_IMG_END_NAMESPACE
#endif