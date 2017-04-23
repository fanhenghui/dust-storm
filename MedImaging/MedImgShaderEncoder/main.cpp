#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>


namespace
{
    static const std::string S_TAG_ENCODER_DST = "EncoderDst";
    static const std::string S_TAG_ECODER_ITEM = "EncoderItem";
}
void main(int argc , char* argv[])
{
    if (argc < 2)
    {
        std::cout <<"Parameter valid!\n";
        return;
    }

    //1 Parse config file
    const char* config_file = argv[1];
    std::fstream in_config(config_file , std::ios::in);
    if (!in_config.is_open())
    {
        std::cout <<"Cant open config file " << in_config << std::endl;
        return;
    }

    std::string line;
    std::ofstream out_encod_dst_file;
    bool is_dst_file_valid = false;
    while(std::getline(in_config , line))
    {
        if (line.empty())
        {
            continue;
        }

        std::stringstream ss(line);
        std::string tag;
        std::string equal_symbol;
        std::string context0;
        std::string context1;
        ss >> tag ;
        if (tag == S_TAG_ENCODER_DST)//Begin a encoding to file sContext0
        {
            ss >> equal_symbol >> context0;
            out_encod_dst_file.clear();
            out_encod_dst_file.close();
            is_dst_file_valid = false;
            out_encod_dst_file.open(context0 , std::ios::out);
            if (!out_encod_dst_file.is_open())
            {
                std::cout << "Cant open encoding dst file : " << context0<< std::endl;
                continue;
            }
            //Write header
            out_encod_dst_file << "#pragma  once\n\n";
            is_dst_file_valid = true;
        }
        else if (tag == S_TAG_ECODER_ITEM)//Encoder shader file sContext0 to opened dst file
        {
            ss >> equal_symbol >> context0 >> context1;
            if (!is_dst_file_valid)
            {
                continue;
            }

            //Read shader file and write to dst file
            if (context1.empty() || context0.empty())
            {
                continue;
            }

            std::ifstream in_shader(context0.c_str() , std::ios::in);
            if (!in_shader.is_open())
            {
                std::cout << "Cant open shader file " << context0 << std::endl;
                continue;
            }

            //Write item variable
            out_encod_dst_file << "static const char* " << context1 <<" = \"\\\n";

            std::string shader_line;
            while(std::getline(in_shader, shader_line))
            {
                out_encod_dst_file << shader_line <<"\\n" << "\\\n";
            }
            //Write item end;
            out_encod_dst_file << "\";\n\n";
            in_shader.close();
        }
        else//print as annotation
        {
            out_encod_dst_file <<"//" << line << std::endl;
        }
    }
    out_encod_dst_file.clear();
    out_encod_dst_file.close();
    in_config.close();
}