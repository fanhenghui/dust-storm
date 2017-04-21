#include "mi_nodule_set_parser.h"
#include "mi_nodule_set.h"

MED_IMAGING_BEGIN_NAMESPACE

namespace
{
    struct NoduleUnit
    {
        unsigned char pos_x[512];
        unsigned char pos_y[512];
        unsigned char pos_z[512];
        unsigned char diameter[512];
        unsigned char type[512];
    };

}

NoduleSetParser::NoduleSetParser()
{

}

NoduleSetParser::~NoduleSetParser()
{

}

IOStatus NoduleSetParser::load_as_csv(const std::string& file_path , std::shared_ptr<NoduleSet>& nodule_set)
{
    std::fstream out(file_path.c_str() , std::ios::out);


    return IO_SUCCESS;
}

IOStatus NoduleSetParser::save_as_csv(const std::string& file_path , const std::shared_ptr<NoduleSet>& nodule_set)
{
    IO_CHECK_NULL_EXCEPTION(nodule_set);

    std::fstream out(file_path.c_str() , std::ios::out);
    if (!out.is_open())
    {
        return IO_FILE_OPEN_FAILED;
    }
    else
    {
        const std::vector<VOISphere>& nodules = nodule_set->get_nodule_set();
        out << "id,coordX,coordY,coordZ,diameter_mm,Type\n";
        out << std::fixed;
        int id = 0;
        for (auto it = nodules.begin() ; it != nodules.end() ; ++it)
        {
            const VOISphere& voi = *it;
            out << id++  << "," << voi.center.x << "," << voi.center.y << ","
                << voi.center.z << "," << voi.diameter<<","<<voi.name << std::endl;
        }
        out.close();
    }

    return IO_SUCCESS;
}

IOStatus NoduleSetParser::load_as_rsa_binary(const std::string& file_path , const mbedtls_rsa_context& rsa , std::shared_ptr<NoduleSet>& nodule_set)
{
    IO_CHECK_NULL_EXCEPTION(nodule_set);

    return IO_SUCCESS;
}

IOStatus NoduleSetParser::save_as_rsa_binary(const std::string& file_path , const mbedtls_rsa_context& rsa , const std::shared_ptr<NoduleSet>& nodule_set)
{
    IO_CHECK_NULL_EXCEPTION(nodule_set);
    
    return IO_SUCCESS;
}

MED_IMAGING_END_NAMESPACE