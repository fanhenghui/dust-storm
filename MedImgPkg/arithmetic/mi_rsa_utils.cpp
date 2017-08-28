#include "mi_rsa_utils.h"

#include <fstream>

#include "mbedtls/entropy.h"
#include "mbedtls/ctr_drbg.h"
#include "mbedtls/bignum.h"
#include "mbedtls/x509.h"
#include "mbedtls/rsa.h"

#ifdef WIN32

#else
#include <string.h>
#endif

MED_IMG_BEGIN_NAMESPACE 

RSAUtils::RSAUtils()
{

}

RSAUtils::~RSAUtils()
{

}

int RSAUtils::gen_key(mbedtls_rsa_context& rsa , int key_size /*= 2048*/ , int exponent /*= 65537*/)
{
    int ret;
    mbedtls_entropy_context entropy;
    mbedtls_ctr_drbg_context ctr_drbg;
    const char *pers = "rsa_genkey";

    mbedtls_ctr_drbg_init( &ctr_drbg );

    mbedtls_entropy_init( &entropy );
    if( ( ret = mbedtls_ctr_drbg_seed( &ctr_drbg, mbedtls_entropy_func, &entropy,
        (const unsigned char *) pers,
        strlen( pers ) ) ) != 0 )
    {
        //mbedtls_printf( " failed\n  ! mbedtls_ctr_drbg_seed returned %d\n", ret );
        return -1;
    }

    mbedtls_rsa_init( &rsa, MBEDTLS_RSA_PKCS_V15, 0 );

    if( ( ret = mbedtls_rsa_gen_key( &rsa, mbedtls_ctr_drbg_random, &ctr_drbg, key_size,
        exponent ) ) != 0 )
    {
        //mbedtls_printf( " failed\n  ! mbedtls_rsa_gen_key returned %d\n\n", ret );
        return -1;
    }

    return 0;
}

int RSAUtils::key_to_file(const mbedtls_rsa_context& rsa , const std::string& file_public , const std::string& file_private)
{
    FILE *fpub  = nullptr;
    if( ( fpub = fopen( file_public.c_str(), "wb+" ) ) == nullptr )
    {
        //mbedtls_printf( " failed\n  ! could not open rsa_pub.txt for writing\n\n" );
        return -1;
    }

    if( ( mbedtls_mpi_write_file( "N = ", &rsa.N, 16, fpub ) ) != 0 ||
        ( mbedtls_mpi_write_file( "E = ", &rsa.E, 16, fpub ) ) != 0 )
    {
        //mbedtls_printf( " failed\n  ! mbedtls_mpi_write_file returned %d\n\n", ret );
        fclose(fpub);
        return -1;
    }
    fclose(fpub);


    FILE *fpriv = nullptr;
    if( ( fpriv = fopen( file_private.c_str(), "wb+" ) ) == nullptr )
    {
        //mbedtls_printf( " failed\n  ! could not open rsa_priv.txt for writing\n" );
        return -1;
    }

    if( ( mbedtls_mpi_write_file( "N = " , &rsa.N , 16, fpriv ) ) != 0 ||
        ( mbedtls_mpi_write_file( "E = " , &rsa.E , 16, fpriv ) ) != 0 ||
        ( mbedtls_mpi_write_file( "D = " , &rsa.D , 16, fpriv ) ) != 0 ||
        ( mbedtls_mpi_write_file( "P = " , &rsa.P , 16, fpriv ) ) != 0 ||
        ( mbedtls_mpi_write_file( "Q = " , &rsa.Q , 16, fpriv ) ) != 0 ||
        ( mbedtls_mpi_write_file( "DP = ", &rsa.DP, 16, fpriv ) ) != 0 ||
        ( mbedtls_mpi_write_file( "DQ = ", &rsa.DQ, 16, fpriv ) ) != 0 ||
        ( mbedtls_mpi_write_file( "QP = ", &rsa.QP, 16, fpriv ) ) != 0 )
    {
        //mbedtls_printf( " failed\n  ! mbedtls_mpi_write_file returned %d\n\n", ret );
        fclose(fpriv);
        return -1;
    }

    fclose(fpriv);

    return 0;
}

int RSAUtils::entrypt(const mbedtls_rsa_context& rsa , size_t len_context , unsigned char* context ,unsigned char (&output)[512])
{
    int return_val;
    const char *pers = "rsa_encrypt";

    mbedtls_entropy_context entropy;
    mbedtls_ctr_drbg_context ctr_drbg;

    mbedtls_ctr_drbg_init( &ctr_drbg );
    mbedtls_entropy_init( &entropy );

    return_val = mbedtls_ctr_drbg_seed( &ctr_drbg, mbedtls_entropy_func,
        &entropy, (const unsigned char *) pers, strlen( pers ) );

    if (return_val != 0)
    {
        return -1;
    }

    return_val = mbedtls_rsa_pkcs1_encrypt( const_cast<mbedtls_rsa_context*>(&rsa), mbedtls_ctr_drbg_random,
        &ctr_drbg, MBEDTLS_RSA_PUBLIC, len_context , context , output );
    if( return_val != 0 )
    {
        return -1;
    }

    return 0;
}

int RSAUtils::detrypt(const mbedtls_rsa_context& rsa , size_t len_context , unsigned char* context , unsigned char (&output)[1024])
{
    int return_val;
    const char *pers = "rsa_decrypt";

    mbedtls_entropy_context entropy;
    mbedtls_ctr_drbg_context ctr_drbg;

    mbedtls_ctr_drbg_init( &ctr_drbg );
    mbedtls_entropy_init( &entropy );

    return_val = mbedtls_ctr_drbg_seed( &ctr_drbg, mbedtls_entropy_func,
        &entropy, (const unsigned char *) pers, strlen( pers ) );

    if (return_val != 0)
    {
        return -1;
    }

    return_val = mbedtls_rsa_pkcs1_decrypt( const_cast<mbedtls_rsa_context*>(&rsa), mbedtls_ctr_drbg_random,
        &ctr_drbg, MBEDTLS_RSA_PRIVATE, &len_context, context, output, 1024 );
    if( return_val != 0 )
    {
        return -1;
    }

    return 0;
}


int RSAUtils::file_to_private_key(const std::string& file_private , mbedtls_rsa_context& rsa)
{
    mbedtls_entropy_context entropy;
    mbedtls_ctr_drbg_context ctr_drbg;

    mbedtls_rsa_init( &rsa, MBEDTLS_RSA_PKCS_V15, 0 );
    mbedtls_ctr_drbg_init( &ctr_drbg );
    mbedtls_entropy_init( &entropy );

    const char *pers = "rsa_decrypt";
    int  return_val = mbedtls_ctr_drbg_seed( &ctr_drbg, mbedtls_entropy_func,
        &entropy, (const unsigned char *) pers, strlen( pers ) );
    if( return_val != 0 )
    {
        return -1;
    }

    FILE *fpriv  = nullptr;
    if( ( fpriv = fopen( "rsa_priv.txt", "rb" ) ) == NULL )
    {
        return -1;
    }

    if( ( return_val = mbedtls_mpi_read_file( &rsa.N , 16, fpriv ) ) != 0 ||
        ( return_val = mbedtls_mpi_read_file( &rsa.E , 16, fpriv ) ) != 0 ||
        ( return_val = mbedtls_mpi_read_file( &rsa.D , 16, fpriv ) ) != 0 ||
        ( return_val = mbedtls_mpi_read_file( &rsa.P , 16, fpriv ) ) != 0 ||
        ( return_val = mbedtls_mpi_read_file( &rsa.Q , 16, fpriv ) ) != 0 ||
        ( return_val = mbedtls_mpi_read_file( &rsa.DP, 16, fpriv ) ) != 0 ||
        ( return_val = mbedtls_mpi_read_file( &rsa.DQ, 16, fpriv ) ) != 0 ||
        ( return_val = mbedtls_mpi_read_file( &rsa.QP, 16, fpriv ) ) != 0 )
    {
        fclose( fpriv );
        return -1;
    }

    rsa.len = ( mbedtls_mpi_bitlen( &rsa.N ) + 7 ) >> 3;

    fclose( fpriv );

    return 0;
}

int RSAUtils::file_to_pubilc_key(const std::string& file_public , mbedtls_rsa_context& rsa)
{
    mbedtls_entropy_context entropy;
    mbedtls_ctr_drbg_context ctr_drbg;

    mbedtls_rsa_init( &rsa, MBEDTLS_RSA_PKCS_V15, 0 );
    mbedtls_ctr_drbg_init( &ctr_drbg );
    mbedtls_entropy_init( &entropy );

    const char *pers = "rsa_encrypt";
    int  return_val = mbedtls_ctr_drbg_seed( &ctr_drbg, mbedtls_entropy_func,
        &entropy, (const unsigned char *) pers, strlen( pers ) );
    if( return_val != 0 )
    {
        return -1;
    }

    FILE *fpub  = nullptr;
    if( ( fpub = fopen( file_public.c_str(), "rb" ) ) == NULL )
    {
        return -1;
    }

    if( ( mbedtls_mpi_read_file( &rsa.N, 16, fpub ) ) != 0 ||
        ( mbedtls_mpi_read_file( &rsa.E, 16, fpub ) ) != 0 )
    {
        fclose(fpub);
        return -1;
    }

    rsa.len = ( mbedtls_mpi_bitlen( &rsa.N ) + 7 ) >> 3;

    fclose( fpub );

    return 0;
}

int RSAUtils::to_rsa_context( 
    const char* n , 
    const char* e , 
    const char* d , 
    const char* p , 
    const char* q , 
    const char* dp , 
    const char* dq , 
    const char* qp ,
    mbedtls_rsa_context& rsa)
{
    mbedtls_rsa_init( &rsa, MBEDTLS_RSA_PKCS_V15, 0 );

    int status = -1;
    if( ( status = mbedtls_mpi_read_string( &rsa.N , 16, n ) ) != 0 ||
        ( status = mbedtls_mpi_read_string( &rsa.E , 16, e ) ) != 0 ||
        ( status = mbedtls_mpi_read_string( &rsa.D , 16, d ) ) != 0 ||
        ( status = mbedtls_mpi_read_string( &rsa.P , 16, p ) ) != 0 ||
        ( status = mbedtls_mpi_read_string( &rsa.Q , 16, q ) ) != 0 ||
        ( status = mbedtls_mpi_read_string( &rsa.DP, 16, dp ) ) != 0 ||
        ( status = mbedtls_mpi_read_string( &rsa.DQ, 16, dq ) ) != 0 ||
        ( status = mbedtls_mpi_read_string( &rsa.QP, 16, qp ) ) != 0 )
    {
        return -1;
    }

    rsa.len = ( mbedtls_mpi_bitlen( &rsa.N ) + 7 ) >> 3;

    return 0;
}


MED_IMG_END_NAMESPACE