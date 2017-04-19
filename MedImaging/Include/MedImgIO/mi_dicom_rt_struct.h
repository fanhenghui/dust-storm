#ifndef MED_IMAGING_RT_STRUCT_H
#define MED_IMAGING_RT_STRUCT_H

#include "MedImgIO/mi_io_stdafx.h"
#include "MedImgArithmetic/mi_point3.h"
#include "MedImgArithmetic/mi_vector3f.h"

MED_IMAGING_BEGIN_NAMESPACE

struct ContourData
{
    std::vector<Vector3f> m_vecPoints;
};

class IO_Export RTStruct
{
public:
    RTStruct();
    ~RTStruct();
    void AddContour(const std::string& sROIName , ContourData* pContour);
    const std::map<std::string , std::vector<ContourData*>>& GetAllContour() const;
    void WriteToFile(const std::string& sFile);
protected:
private:
    std::map<std::string , std::vector<ContourData*>> m_mapRTData;
};

MED_IMAGING_END_NAMESPACE

#endif