#include "mi_nodule_set_parser.h"
#include "mi_nodule_set.h"

#include "MedImgArithmetic/mi_rsa_utils.h"
#include "MedImgCommon/mi_string_number_converter.h"

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

        NoduleUnit()
        {
            memset(pos_x , 0 , sizeof(pos_x));
            memset(pos_y , 0 , sizeof(pos_y));
            memset(pos_z , 0 , sizeof(pos_z));
            memset(diameter , 0 , sizeof(diameter));
            memset(type , 0 , sizeof(type));
        }
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
    nodule_set->clear_nodule();

    std::fstream in(file_path , std::ios::in | std::ios::binary);
    if (!in.is_open())
    {
        return IO_FILE_OPEN_FAILED;
    }

    RSAUtils rsa_utils;
    StrNumConverter<double> str_num_convertor;
    int status(0);

    //1 Read nodule number
    unsigned char buffer[1024];
    unsigned char input_nudule_num[512];
    if(!in.read((char*)input_nudule_num , sizeof(input_nudule_num)))
    {
        in.close();
        return IO_DATA_DAMAGE;
    }

    memset(buffer , 0 , sizeof(buffer));
    status = rsa_utils.detrypt(rsa , 512 ,input_nudule_num , buffer );
    if (status != 0)
    {
        in.close();
        return IO_ENCRYPT_FAILED;
    }

    const int num =  static_cast<int>(str_num_convertor.to_num(std::string((char*)buffer)));
    if (num < 0)//warning no nodule
    {
        in.close();
        return IO_SUCCESS;
    }

    for (int i = 0 ; i < num ; ++i)
    {
        NoduleUnit nodule_unit;
        if (!in.read((char*)(&nodule_unit) , sizeof(nodule_unit)))
        {
            break;
        }

        memset(buffer , 0 , sizeof(buffer));
        status = rsa_utils.detrypt(rsa , 512 ,nodule_unit.pos_x , buffer );
        if (status != 0)
        {
            in.close();
            return IO_ENCRYPT_FAILED;
        }
        double pos_x=  static_cast<int>(str_num_convertor.to_num(std::string((char*)buffer)));

        memset(buffer , 0 , sizeof(buffer));
        status = rsa_utils.detrypt(rsa , 512 ,nodule_unit.pos_y , buffer );
        if (status != 0)
        {
            in.close();
            return IO_ENCRYPT_FAILED;
        }
        double pos_y=  static_cast<int>(str_num_convertor.to_num(std::string((char*)buffer)));

        memset(buffer , 0 , sizeof(buffer));
        status = rsa_utils.detrypt(rsa , 512 ,nodule_unit.pos_z , buffer );
        if (status != 0)
        {
            in.close();
            return IO_ENCRYPT_FAILED;
        }
        double pos_z=  static_cast<int>(str_num_convertor.to_num(std::string((char*)buffer)));

        memset(buffer , 0 , sizeof(buffer));
        status = rsa_utils.detrypt(rsa , 512 ,nodule_unit.diameter , buffer );
        if (status != 0)
        {
            in.close();
            return IO_ENCRYPT_FAILED;
        }
        double diameter=  static_cast<int>(str_num_convertor.to_num(std::string((char*)buffer)));

        memset(buffer , 0 , sizeof(buffer));
        status = rsa_utils.detrypt(rsa , 512 ,nodule_unit.type , buffer );
        if (status != 0)
        {
            in.close();
            return IO_ENCRYPT_FAILED;
        }
        std::string type =  std::string((char*)buffer);

        nodule_set->add_nodule(VOISphere(Point3(pos_x , pos_y , pos_z) , diameter , type));

    }

    in.close();
    return IO_SUCCESS;
}

IOStatus NoduleSetParser::save_as_rsa_binary(const std::string& file_path , const mbedtls_rsa_context& rsa , const std::shared_ptr<NoduleSet>& nodule_set)
{
    IO_CHECK_NULL_EXCEPTION(nodule_set);

    std::fstream out(file_path, std::ios::out | std::ios::binary);
    if (!out.is_open())
    {
        return IO_FILE_OPEN_FAILED;
    }

    RSAUtils rsa_utils;
    StrNumConverter<double> str_num_convertor;
    int status(0);

    //1 Write nodule number
    unsigned char output_nudule_num[512];
    memset(output_nudule_num , 0 ,  sizeof(output_nudule_num));

    const std::vector<VOISphere>& nodules = nodule_set->get_nodule_set();
    std::string nodule_num = str_num_convertor.to_string(static_cast<double>(nodules.size()));

    status = rsa_utils.entrypt(rsa , nodule_num.size() , (unsigned char*)(nodule_num.c_str()) , output_nudule_num );
    if (status != 0)
    {
        out.close();
        return IO_ENCRYPT_FAILED;
    }

    out.write((char*)output_nudule_num , sizeof(output_nudule_num));

    //2 Save nodule number
    int id = 0;
    for (auto it = nodules.begin() ; it != nodules.end() ; ++it)
    {
        const VOISphere& voi = *it;
        std::string pos_x = str_num_convertor.to_string(voi.center.x);
        std::string pos_y = str_num_convertor.to_string(voi.center.y);
        std::string pos_z = str_num_convertor.to_string(voi.center.z);
        std::string diameter = str_num_convertor.to_string(voi.diameter);

        NoduleUnit nodule_unit;

        status = rsa_utils.entrypt(rsa , pos_x.size() , (unsigned char*)(pos_x.c_str()) , nodule_unit.pos_x );
        if (status != 0)
        {
            out.close();
            return IO_ENCRYPT_FAILED;
        }

        status = rsa_utils.entrypt(rsa , pos_y.size() , (unsigned char*)(pos_y.c_str()) , nodule_unit.pos_y );
        if (status != 0)
        {
            out.close();
            return IO_ENCRYPT_FAILED;
        }

        status = rsa_utils.entrypt(rsa , pos_z.size() , (unsigned char*)(pos_z.c_str()) , nodule_unit.pos_z );
        if (status != 0)
        {
            out.close();
            return IO_ENCRYPT_FAILED;
        }

        status = rsa_utils.entrypt(rsa , diameter.size() , (unsigned char*)(diameter.c_str()) , nodule_unit.diameter );
        if (status != 0)
        {
            out.close();
            return IO_ENCRYPT_FAILED;
        }

        status = rsa_utils.entrypt(rsa , voi.name.size() , (unsigned char*)(voi.name.c_str()) , nodule_unit.type );
        if (status != 0)
        {
            out.close();
            return IO_ENCRYPT_FAILED;
        }

        out.write((char*)(&nodule_unit) , sizeof(nodule_unit));
    }
    out.close();

    return IO_SUCCESS;
}

MED_IMAGING_END_NAMESPACE