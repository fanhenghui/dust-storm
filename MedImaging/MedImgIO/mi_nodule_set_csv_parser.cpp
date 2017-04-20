#include "mi_nodule_set_csv_parser.h"
#include "mi_nodule_set.h"

MED_IMAGING_BEGIN_NAMESPACE

NoduleSetCSVParser::NoduleSetCSVParser()
{

}

NoduleSetCSVParser::~NoduleSetCSVParser()
{

}

IOStatus NoduleSetCSVParser::load(const std::string& sFilePath , std::shared_ptr<NoduleSet>& pNoduleSet)
{
    std::fstream out(sFilePath.c_str() , std::ios::out);


    return IO_SUCCESS;
}

IOStatus NoduleSetCSVParser::save(const std::string& sFilePath , const std::shared_ptr<NoduleSet>& pNoduleSet)
{
    IO_CHECK_NULL_EXCEPTION(pNoduleSet);

    std::fstream out(sFilePath.c_str() , std::ios::out);
    if (!out.is_open())
    {
        return IO_FILE_OPEN_FAILED;
    }
    else
    {
        const std::vector<VOISphere>& nodules = pNoduleSet->get_nodule_set();
        out << "id,coordX,coordY,coordZ,diameter_mm,Type\n";
        out << std::fixed;
        int id = 0;
        for (auto it = nodules.begin() ; it != nodules.end() ; ++it)
        {
            const VOISphere& voi = *it;
            out << id++  << "," << voi.m_ptCenter.x << "," << voi.m_ptCenter.y << ","
                << voi.m_ptCenter.z << "," << voi.m_dDiameter<<","<<voi.m_sName << std::endl;
        }
        out.close();
    }

    return IO_SUCCESS;
}

MED_IMAGING_END_NAMESPACE