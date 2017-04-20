#include "mi_dicom_rt_struct.h"

MED_IMAGING_BEGIN_NAMESPACE

RTStruct::RTStruct()
{

}

RTStruct::~RTStruct()
{

}

void RTStruct::add_contour(const std::string& sROIName , ContourData* pContour)
{
    IO_CHECK_NULL_EXCEPTION(pContour);

    if (m_mapRTData.find(sROIName) == m_mapRTData.end())
    {
        std::vector<ContourData*> v;
        v.push_back(pContour);
        m_mapRTData[sROIName] = v;
    }
    else
    {
        m_mapRTData[sROIName].push_back(pContour);
    }
}

const std::map<std::string , std::vector<ContourData*>>& RTStruct::get_all_contour() const
{
    return m_mapRTData;
}

void RTStruct::write_to_file(const std::string& sFile)
{
    std::ofstream out(sFile , std::ios::out);
    if (out.is_open())
    {
        out << "RT Structure set\n";
        out << "ROI number : " << m_mapRTData.size() << std::endl;
        out << "ROI list : [Name] [Contour number] \n";
        for (auto it = m_mapRTData.begin() ; it != m_mapRTData.end() ; ++it)
        {
            out << it->first << " " << it->second.size() << std::endl;
        }

        out << "Contour list : \n";
        
        for (auto it = m_mapRTData.begin() ; it != m_mapRTData.end() ; ++it)
        {
            out << "ROI : " << it->first  <<" 's contour list : "<< std::endl;
            int iROICount =0;
            for (auto it2 = it->second.begin() ; it2 != it->second.end() ; ++it2)
            {
                out << "Contour " << iROICount << " begin : "<< std::endl;
                for (auto it3 = (*it2)->m_vecPoints.begin() ; it3 != (*it2)->m_vecPoints.end() ; ++it3)
                {
                    out << (*it3)._m[0] <<" "<<(*it3)._m[1] << " " << (*it3)._m[2] << std::endl;
                }
                out << "Contour " << iROICount++ << " end. "<< std::endl;
            }
        }

        out.close();
    }
}

MED_IMAGING_END_NAMESPACE