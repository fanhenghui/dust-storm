#ifndef MED_IMG_DICOM_RT_STRUCT_SET_LOADER_H
#define MED_IMG_DICOM_RT_STRUCT_SET_LOADER_H

#include <memory>
#include "io/mi_io_export.h"
#include "io/mi_io_define.h"

class DcmFileFormat;
class DcmDataset;
class DcmMetaInfo;

typedef std::shared_ptr<DcmFileFormat> DcmFileFormatPtr;

MED_IMG_BEGIN_NAMESPACE 
class RTStruct;
class IO_Export DICOMRTLoader
{
public:
    DICOMRTLoader();
    ~DICOMRTLoader();
    IOStatus load_rt_struct(const std::string& file_name , std::shared_ptr<RTStruct> &pRTStruct);
    IOStatus load_rt_struct(DcmFileFormatPtr pFileFormat , std::shared_ptr<RTStruct> &pRTStruct);
protected:
private:
};

MED_IMG_END_NAMESPACE


#endif