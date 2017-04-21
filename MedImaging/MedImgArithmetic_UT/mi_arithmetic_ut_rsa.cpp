#include <iostream>
#include "MedImgArithmetic/mi_rsa_utils.h"


using namespace medical_imaging;

void UT_RSA()
{
    int status = 0;
    mbedtls_rsa_context rsa;

    RSAUtils rsa_tools;

    status = rsa_tools.gen_key(rsa);

    if(status != 0)
    {
        std::cout << "Generate key failed!\n";
    }

    rsa_tools.key_to_file(rsa , "D:/temp/rsa_pub.txt" , "D:/temp/rsa_pri.txt");

    std::string pos_x("45.62359");
    std::string diameter("0.2663154");
    std::string nodule_type("ACC");

    unsigned char entrypt_output_pos_x[512];
    unsigned char entrypt_output_diameter[512];
    unsigned char entrypt_output_nodule_type[512];

    status = rsa_tools.entrypt(rsa , pos_x.size() ,(unsigned char*)(pos_x.c_str()) , entrypt_output_pos_x);
    if(status != 0)
    {
        std::cout << "entrypt failed!\n";
    }

    status = rsa_tools.entrypt(rsa , diameter.size() ,(unsigned char*)(diameter.c_str()) , entrypt_output_diameter);
    if(status != 0)
    {
        std::cout << "entrypt failed!\n";
    }

    status = rsa_tools.entrypt(rsa , nodule_type.size() , (unsigned char*)(nodule_type.c_str()) , entrypt_output_nodule_type);
    if(status != 0)
    {
        std::cout << "entrypt failed!\n";
    }

    unsigned char detrypt_output_pos_x[1024];
    unsigned char detrypt_output_diameter[1024];
    unsigned char detrypt_output_nodule_type[1024];

    memset(detrypt_output_pos_x , 0 , sizeof(detrypt_output_pos_x));
    memset(detrypt_output_diameter , 0 , sizeof(detrypt_output_diameter));
    memset(detrypt_output_nodule_type , 0 , sizeof(detrypt_output_nodule_type));


    status = rsa_tools.detrypt(rsa ,  512 , entrypt_output_pos_x , detrypt_output_pos_x);
    if(status != 0)
    {
        std::cout << "detrypt failed!\n";
    }
    status = rsa_tools.detrypt(rsa , 512 , entrypt_output_diameter , detrypt_output_diameter);
    if(status != 0)
    {
        std::cout << "detrypt failed!\n";
    }
    status = rsa_tools.detrypt(rsa , 512 , entrypt_output_nodule_type , detrypt_output_nodule_type);
    if(status != 0)
    {
        std::cout << "detrypt failed!\n";
    }

    std::string result_pos_x((char*)detrypt_output_pos_x);
    std::string result_diameter((char*)detrypt_output_diameter);
    std::string result_nodule_type((char*)detrypt_output_nodule_type);

    std::cout << result_pos_x << "\n";
    std::cout << result_diameter << "\n";
    std::cout << result_nodule_type << "\n";

}