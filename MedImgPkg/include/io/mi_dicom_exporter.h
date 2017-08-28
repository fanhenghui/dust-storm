#ifndef MED_IMG_DICOM_EXPORTER_H
#define MED_IMG_DICOM_EXPORTER_H

#include <memory>

#include "io/mi_io_export.h"
#include "io/mi_io_define.h"

#include "dcmtk/dcmdata/dcdeftag.h"

class DcmFileFormat;
class DcmDataset;
class DcmMetaInfo;
class DcmTagKey;

typedef std::shared_ptr<DcmFileFormat> DcmFileFormatPtr;
typedef std::vector<DcmFileFormatPtr> DcmFileFormatSet;

MED_IMG_BEGIN_NAMESPACE 

class ImageData;
class ImageDataHeader;
class ProgressModel;

class DICOMExporter
{
public:
    IO_Export DICOMExporter();
    IO_Export ~DICOMExporter();
    IO_Export IOStatus export_series(const std::vector<std::string>& in_files , 
        const std::vector<std::string>& out_files, ExportDicomDataType etype);
    IO_Export void set_progress_model(std::shared_ptr<ProgressModel> model);
    IO_Export void set_anonymous_taglist(const std::vector<DcmTagKey> &tag_list);
    IO_Export void skip_derived_image(bool flag);

private:
    IOStatus load_dicom_file(const std::string file_name , DcmFileFormatPtr&);
    IOStatus save_dicom_as_bitmap(const std::string in_files, const std::string out_files);
    void anonymous_dicom_data(DcmFileFormatPtr fileformat_ptr);
    void anonymous_all_patient_name(DcmFileFormatPtr fileformat_ptr);
    void remove_private_tag(DcmFileFormatPtr fileformat_ptr);
    IOStatus save_dicom_as_raw(const std::string in_file_name, const std::string out_file_name);////////////////////////////////////

private:

    void add_progress_i(float value);
    void set_progress_i(int value);

    float _progress;
    std::shared_ptr<ProgressModel> _model;
    std::vector<DcmTagKey> _taglist;
    bool _skip_derived_image;
};

MED_IMG_END_NAMESPACE
#endif