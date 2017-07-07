#ifndef MED_IMAGING_RT_STRUCT_H
#define MED_IMAGING_RT_STRUCT_H

#include "MedImgIO/mi_io_stdafx.h"
#include "MedImgArithmetic/mi_point3.h"
#include "MedImgArithmetic/mi_vector3f.h"

#include <vector>

MED_IMAGING_BEGIN_NAMESPACE

struct ContourData
{
    std::vector<Vector3f> points;
};

class IO_Export RTStruct
{
public:
    RTStruct();
    ~RTStruct();
    void add_contour(const std::string& roi_name , ContourData* contour);
    const std::map<std::string , std::vector<ContourData*>>& get_all_contour() const;
    void write_to_file(const std::string& file_name);
protected:
private:
    std::map<std::string , std::vector<ContourData*>> rt_data;
};

MED_IMAGING_END_NAMESPACE

#endif