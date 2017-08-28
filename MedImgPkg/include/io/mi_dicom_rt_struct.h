#ifndef MED_IMG_RT_STRUCT_H
#define MED_IMG_RT_STRUCT_H

#include "io/mi_io_export.h"
#include "arithmetic/mi_point3.h"
#include "arithmetic/mi_vector3f.h"

MED_IMG_BEGIN_NAMESPACE

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

MED_IMG_END_NAMESPACE

#endif