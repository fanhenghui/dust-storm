#ifndef MED_IMAGING_RSA_UTILS_H
#define MED_IMAGING_RSA_UTILS_H

#include "MedImgArithmetic/mi_arithmetic_stdafx.h"

#include "mbedtls/config.h"
#include "mbedtls/platform.h"
#include "mbedtls/rsa.h"

MED_IMAGING_BEGIN_NAMESPACE

class Arithmetic_Export RSAUtils
{
public:
    RSAUtils();
    ~RSAUtils();

    int gen_key(mbedtls_rsa_context& rsa , int key_size = 2048 , int exponent = 65537);

    int key_to_file(const mbedtls_rsa_context& rsa , const std::string& file_public , const std::string& file_private);
    int file_to_pubilc_key(const std::string& file_public , mbedtls_rsa_context& rsa );
    int file_to_private_key(const std::string& file_private , mbedtls_rsa_context& rsa );

    int entrypt(const mbedtls_rsa_context& rsa , size_t len_context , unsigned char* context ,unsigned char (&output)[512] );
    int detrypt(const mbedtls_rsa_context& rsa , size_t len_context , unsigned char* context , unsigned char (&output)[1024] );

protected:
private:
};

MED_IMAGING_END_NAMESPACE

#endif