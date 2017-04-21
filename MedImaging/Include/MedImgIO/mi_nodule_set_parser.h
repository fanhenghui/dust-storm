#ifndef MED_IMAGING_NODULE_SET_CSV_PARSER_H_
#define MED_IMAGING_NODULE_SET_CSV_PARSER_H_

#include "MedImgIO/mi_io_stdafx.h"
#include "MedImgCommon/mi_common_define.h"
#include "mbedtls/rsa.h"

MED_IMAGING_BEGIN_NAMESPACE

class NoduleSet;
class IO_Export NoduleSetParser
{
public:
    NoduleSetParser();
    ~NoduleSetParser();

    IOStatus load_as_csv(const std::string& file_path , std::shared_ptr<NoduleSet>& nodule_set);
    IOStatus save_as_csv(const std::string& file_path , const std::shared_ptr<NoduleSet>& nodule_set);

    IOStatus load_as_rsa_binary(const std::string& file_path , const mbedtls_rsa_context& rsa , std::shared_ptr<NoduleSet>& nodule_set);
    IOStatus save_as_rsa_binary(const std::string& file_path , const mbedtls_rsa_context& rsa , const std::shared_ptr<NoduleSet>& nodule_set);

protected:
private:
};

MED_IMAGING_END_NAMESPACE

#endif