#ifndef MED_IMAGING_DICOM_RT_STRUCT_SET_LOADER_H
#define MED_IMAGING_DICOM_RT_STRUCT_SET_LOADER_H

#include "MedImgIO/mi_io_stdafx.h"
#include "MedImgCommon/mi_common_define.h"

class DcmFileFormat;
class DcmDataset;
class DcmMetaInfo;

typedef std::shared_ptr<DcmFileFormat> DcmFileFormatPtr;

MED_IMAGING_BEGIN_NAMESPACE
class RTStruct;
class IO_Export DICOMRTLoader
{
public:
    DICOMRTLoader();
    ~DICOMRTLoader();
    IOStatus load_rt_struct(const std::string& sFile , std::shared_ptr<RTStruct> &pRTStruct);
    IOStatus load_rt_struct(DcmFileFormatPtr pFileFormat , std::shared_ptr<RTStruct> &pRTStruct);
protected:
private:
};

MED_IMAGING_END_NAMESPACE


#endif