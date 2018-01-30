#ifndef MEDIMGIO_DICOM_LOADER_H
#define MEDIMGIO_DICOM_LOADER_H

#include "arithmetic/mi_point3.h"
#include "arithmetic/mi_vector3.h"
#include "arithmetic/mi_point2.h"
#include "io/mi_io_define.h"
#include "io/mi_io_export.h"

class DcmFileFormat;
class DcmDataset;
class DcmMetaInfo;

typedef std::shared_ptr<DcmFileFormat> DcmFileFormatPtr;
typedef std::vector<DcmFileFormatPtr> DcmFileFormatSet;

MED_IMG_BEGIN_NAMESPACE

class ImageData;
class ImageDataHeader;
class ProgressModel;
class DICOMLoader {
public:
    struct DCMSliceStream {
        char* buffer;
        unsigned int size;
        bool ref;
        DCMSliceStream(char* buffer_, int size_, bool ref_=false):
            buffer(buffer_),size(size_),ref(ref_) {}
        ~DCMSliceStream() {
            if (!ref && buffer) {
                delete [] buffer;
                buffer = nullptr;
            }
        }
    };
public:
    IO_Export DICOMLoader();

    IO_Export ~DICOMLoader();

    IO_Export IOStatus load_dicom(std::string& dcm_file, std::shared_ptr<ImageDataHeader>& img_data_header);

    // files will removal invalid file and keep majority series
    IO_Export IOStatus load_series(std::vector<std::string>& files_in_out,
                std::shared_ptr<ImageData>& image_data,
                std::shared_ptr<ImageDataHeader>& img_data_header);

    
    IO_Export IOStatus load_series(std::vector<DCMSliceStream*>& buffers,
                std::shared_ptr<ImageData>& image_data,
                std::shared_ptr<ImageDataHeader>& img_data_header);

    IO_Export void set_progress_model(std::shared_ptr<ProgressModel> model);

    IO_Export IOStatus check_series_uid(const std::string& file,
                                        std::string& study_uid,
                                        std::string& series_uid);

    IO_Export IOStatus check_series_uid(const DCMSliceStream* stream,
                                        std::string& study_uid,
                                        std::string& series_uid);

    IO_Export IOStatus check_series_uid(const std::string& file,
                                        std::string& study_uid,
                                        std::string& series_uid,
                                        std::string& patient_name,
                                        std::string& patient_id,
                                        std::string& modality);

    IO_Export IOStatus check_series_uid(const DCMSliceStream* stream,
                                        std::string& study_uid,
                                        std::string& series_uid,
                                        std::string& patient_name,
                                        std::string& patient_id,
                                        std::string& modality);

private:
    IOStatus data_check(std::vector<std::string>& files,
                          DcmFileFormatSet& file_format_set);
    IOStatus data_check_series_unique(std::vector<std::string>& files,
                                        DcmFileFormatSet& file_format_set);
    IOStatus data_check_spacing_unique(DcmFileFormatSet& set_in, 
                                         DcmFileFormatSet& set_out);
    /*IOStatus data_check_series_image_sequece_i(DcmFileFormatSet& set_in, 
                                               DcmFileFormatSet& set_out);*/

    IOStatus sort_series(DcmFileFormatSet& file_format_set);

    IOStatus construct_data_header(DcmFileFormatPtr& file_format,
                            std::shared_ptr<ImageDataHeader> img_data_header);

    IOStatus construct_image_data(DcmFileFormatSet& file_format_set,
                           std::shared_ptr<ImageDataHeader> img_data_header,
                           std::shared_ptr<ImageData> image_data);

private:
    bool get_transfer_syntax_uid(DcmMetaInfo* meta_info,
                              std::shared_ptr<ImageDataHeader>& img_data_header);

    bool get_content_time(DcmDataset* data_set,
                            std::shared_ptr<ImageDataHeader>& img_data_header);

    bool get_manufacturer(DcmDataset* data_set,
                            std::shared_ptr<ImageDataHeader>& img_data_header);

    bool get_manufacturer_model_name(
        DcmDataset* data_set, std::shared_ptr<ImageDataHeader>& img_data_header);

    bool get_patient_name(DcmDataset* data_set,
                            std::shared_ptr<ImageDataHeader>& img_data_header);

    bool get_patient_id(DcmDataset* data_set,
                          std::shared_ptr<ImageDataHeader>& img_data_header);

    bool get_patient_sex(DcmDataset* data_set,
                           std::shared_ptr<ImageDataHeader>& img_data_header);

    bool get_patient_age(DcmDataset* data_set,
                           std::shared_ptr<ImageDataHeader>& img_data_header);

    bool get_slice_thickness(DcmDataset* data_set,
                               std::shared_ptr<ImageDataHeader>& img_data_header);

    bool get_kvp(DcmDataset* data_set,
                   std::shared_ptr<ImageDataHeader>& img_data_header);

    bool get_patient_position(DcmDataset* data_set,
                           std::shared_ptr<ImageDataHeader>& img_data_header);

    bool get_series_uid(DcmDataset* data_set,
                          std::shared_ptr<ImageDataHeader>& img_data_header);

    bool get_study_uid(DcmDataset* data_set,
                         std::shared_ptr<ImageDataHeader>& img_data_header);

    bool get_sop_instance_uid(DcmDataset* data_set,
                         std::shared_ptr<ImageDataHeader>& img_data_header);

    bool get_sample_per_pixel(DcmDataset* data_set,
                           std::shared_ptr<ImageDataHeader>& img_data_header);

    bool get_rows(DcmDataset* data_set,
                    std::shared_ptr<ImageDataHeader>& img_data_header);

    bool get_columns(DcmDataset* data_set,
                       std::shared_ptr<ImageDataHeader>& img_data_header);

    bool get_pixel_spacing(DcmDataset* data_set,
                             std::shared_ptr<ImageDataHeader>& img_data_header);
    bool get_pixel_spacing(DcmDataset* data_set, Point2& spacing);

    bool get_bits_allocated(DcmDataset* data_set,
                              std::shared_ptr<ImageDataHeader>& img_data_header);

    bool get_pixel_representation(DcmDataset* data_set,
                               std::shared_ptr<ImageDataHeader>& img_data_header);

    bool get_intercept(DcmDataset* data_set, float& intercept);

    bool get_slope(DcmDataset* data_set, float& slope);

    bool get_instance_number(DcmDataset* data_set, int& instance_num);

    bool get_image_position(DcmDataset* data_set, Point3& image_position);

    bool get_image_orientation(DcmDataset* data_set, Vector3& row_orientation,
                                 Vector3& column_orientation);

    bool get_slice_location(DcmDataset* data_set, double& slice_location);

    bool get_pixel_data(DcmFileFormatPtr pFileFormat, DcmDataset* data_set,
                          char* data_array, unsigned int length);

    bool get_jpeg_compressed_pixel_data(DcmFileFormatPtr pFileFormat,
                                          DcmDataset* data_set, char* data_array,
                                          unsigned int length);

private:
    void add_progress(float value);
    void set_progress(int value);

    float _progress;
    std::shared_ptr<ProgressModel> _model;
};

MED_IMG_END_NAMESPACE
#endif