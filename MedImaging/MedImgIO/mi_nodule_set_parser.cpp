#include "mi_nodule_set_parser.h"

#include "boost/format.hpp"
#include "boost/tokenizer.hpp"
#include "boost/algorithm/string.hpp"  

#include "MedImgArithmetic/mi_rsa_utils.h"
#include "MedImgCommon/mi_string_number_converter.h"

#include "mi_nodule_set.h"

MED_IMAGING_BEGIN_NAMESPACE

namespace
{
    struct NoduleUnit
    {
        int nodule_id;//none encoded
        unsigned char series_id[512];//none encoded
        unsigned char pos_x[512];
        unsigned char pos_y[512];
        unsigned char pos_z[512];
        unsigned char diameter[512];
        unsigned char type[512];

        NoduleUnit()
        {
            nodule_id = 0;
            memset(series_id , 0 , sizeof(series_id));
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
    std::fstream in(file_path.c_str() , std::ios::in);

    if (!in.is_open())
    {
        return IO_FILE_OPEN_FAILED;
    }

    nodule_set->clear_nodule();
    //series uid,nodule uid,coordX,coordY,coordZ,diameter_mm,Type
    std::string line;
    std::getline(in , line);
    std::vector<std::string> infos;
    boost::split(infos , line , boost::is_any_of(","));
    const size_t nodule_file_type = infos.size();
    if (nodule_file_type != 7 && nodule_file_type != 5)
    {
        in.close();
        return IO_UNSUPPORTED_YET;
    }


    std::string series_id;
    double id , pos_x , pos_y , pos_z , diameter;
    std::string separate;
    std::string type;
    StrNumConverter<double> str_num_converter;
    while(std::getline(in , line))
    {
        std::vector<std::string> infos;
        boost::split(infos , line , boost::is_any_of(","));
        if (infos.size() != nodule_file_type)
        {
            in.close();
            return IO_DATA_DAMAGE;
        }
        if (7 == nodule_file_type)//Mine nodule list file
        {
            series_id = infos[0];

            //Check series id
            if (!_series_id.empty())
            {
                if(series_id != _series_id)
                {
                    return IO_UNMATCHED_FILE;
                }
            }

            id = str_num_converter.to_num(infos[1]);
            pos_x = str_num_converter.to_num(infos[2]);
            pos_y = str_num_converter.to_num(infos[3]);
            pos_z = str_num_converter.to_num(infos[4]);
            diameter = str_num_converter.to_num(infos[5]);
            type = infos[6];
            nodule_set->add_nodule(VOISphere(Point3(pos_x , pos_y , pos_z) , diameter , type));
        }
        else if (5 == nodule_file_type)//Luna nodule list file
        {
            series_id = infos[0];

            //Check series id
            if(series_id != _series_id)
            {
                continue;
            }

            pos_x = str_num_converter.to_num(infos[1]);
            pos_y = str_num_converter.to_num(infos[2]);
            pos_z = str_num_converter.to_num(infos[3]);
            diameter = str_num_converter.to_num(infos[4]);
            nodule_set->add_nodule(VOISphere(Point3(pos_x , pos_y , pos_z) , diameter , "W"));
        }
    }

    in.close();

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
        out << "seriesuid,noduleuid,coordX,coordY,coordZ,diameter_mm,Type\n";
        out << std::fixed;
        int id = 0;
        for (auto it = nodules.begin() ; it != nodules.end() ; ++it)
        {
            const VOISphere& voi = *it;
            out << _series_id <<"," << id++  << "," << voi.center.x << "," << voi.center.y << ","
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

        //Check series id
        if (!_series_id.empty())
        {
            if (nodule_unit.series_id[_series_id.size()] != '\0')
            {
                return IO_UNMATCHED_FILE;
            }

            int i = 0;
            for (; i < 512 && i<_series_id.size() ; ++i)
            {
                if (nodule_unit.series_id[i] == '\0' || nodule_unit.series_id[i] != _series_id[i])
                {
                    break;
                }
            }
            if (i != _series_id.size())
            {
                return IO_UNMATCHED_FILE;
            }
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

        nodule_unit.nodule_id = id++;
        if (!_series_id.empty())
        {
            if (_series_id.size() > 511)
            {
                return IO_UNSUPPORTED_YET;
            }
            for (int i = 0 ; i<_series_id.size() ; ++i)
            {
                nodule_unit.series_id[i] = _series_id[i];
            }
            nodule_unit.series_id[_series_id.size()] = '\0';
        }

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

void NoduleSetParser::set_series_id(const std::string& series_id)
{
    _series_id = series_id;
}



MED_IMAGING_END_NAMESPACE