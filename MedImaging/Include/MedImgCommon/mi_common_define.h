#ifndef MED_IMAGING_COMMON_DEGINE_H_
#define MED_IMAGING_COMMON_DEGINE_H_

#include "med_imaging_config.h"

MED_IMAGING_BEGIN_NAMESPACE

enum DataType
{
    CHAR = 0,
    UCHAR,
    SHORT,
    USHORT,
    FLOAT,
};

enum ProcessingUnitType
{
    CPU = 0,
    GPU,
};

enum IOStatus
{
    IO_SUCCESS,
    IO_EMPTY_INPUT,
    IO_FILE_OPEN_FAILED,
    IO_DATA_DAMAGE,
    IO_UNSUPPORTED_YET,
    IO_ENCRYPT_FAILED,
    IO_DECRYPT_FAILED,
};

MED_IMAGING_END_NAMESPACE
#endif