#ifndef MED_IMAGING_NODULE_SET_CSV_PARSER_H_
#define MED_IMAGING_NODULE_SET_CSV_PARSER_H_

#include "MedImgIO/mi_io_stdafx.h"
#include "MedImgCommon/mi_common_define.h"

MED_IMAGING_BEGIN_NAMESPACE

class NoduleSet;
class IO_Export NoduleSetCSVParser
{
public:
    NoduleSetCSVParser();
    ~NoduleSetCSVParser();
    IOStatus load(const std::string& file_path , std::shared_ptr<NoduleSet>& nodule_set);
    IOStatus save(const std::string& file_path , const std::shared_ptr<NoduleSet>& nodule_set);
protected:
private:
};

MED_IMAGING_END_NAMESPACE

#endif