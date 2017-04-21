#include "mi_dicom_rt_struct.h"

MED_IMAGING_BEGIN_NAMESPACE

RTStruct::RTStruct()
{

}

RTStruct::~RTStruct()
{

}

void RTStruct::add_contour(const std::string& roi_name , ContourData* contour)
{
    IO_CHECK_NULL_EXCEPTION(contour);

    if (rt_data.find(roi_name) == rt_data.end())
    {
        std::vector<ContourData*> v;
        v.push_back(contour);
        rt_data[roi_name] = v;
    }
    else
    {
        rt_data[roi_name].push_back(contour);
    }
}

const std::map<std::string , std::vector<ContourData*>>& RTStruct::get_all_contour() const
{
    return rt_data;
}

void RTStruct::write_to_file(const std::string& file_name)
{
    std::ofstream out(file_name , std::ios::out);
    if (out.is_open())
    {
        out << "RT Structure set\n";
        out << "ROI number : " << rt_data.size() << std::endl;
        out << "ROI list : [Name] [Contour number] \n";
        for (auto it = rt_data.begin() ; it != rt_data.end() ; ++it)
        {
            out << it->first << " " << it->second.size() << std::endl;
        }

        out << "Contour list : \n";
        
        for (auto it = rt_data.begin() ; it != rt_data.end() ; ++it)
        {
            out << "ROI : " << it->first  <<" 's contour list : "<< std::endl;
            int roi_count =0;
            for (auto it2 = it->second.begin() ; it2 != it->second.end() ; ++it2)
            {
                out << "Contour " << roi_count << " begin : "<< std::endl;
                for (auto it3 = (*it2)->points.begin() ; it3 != (*it2)->points.end() ; ++it3)
                {
                    out << (*it3)._m[0] <<" "<<(*it3)._m[1] << " " << (*it3)._m[2] << std::endl;
                }
                out << "Contour " << roi_count++ << " end. "<< std::endl;
            }
        }

        out.close();
    }
}

MED_IMAGING_END_NAMESPACE