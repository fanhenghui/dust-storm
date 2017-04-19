#ifndef MED_IMAGING_IO_H
#define MED_IMAGING_IO_H

#include "med_imaging_config.h"

#include <exception>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>
#include <cmath>

#include "MedImgCommon/mi_common_exception.h"


MED_IMAGING_BEGIN_NAMESPACE

#ifdef MEDIMGIO_EXPORTS
#define IO_Export __declspec(dllexport)
#else
#define IO_Export __declspec(dllimport)
#endif

#pragma warning(disable: 4251 4819 4099)

#ifndef IO_THROW_EXCEPTION
#define IO_THROW_EXCEPTION(desc) THROW_EXCEPTION("IO" , desc);
#endif

#ifndef IO_CHECK_NULL_EXCEPTION
#define  IO_CHECK_NULL_EXCEPTION(pointer)                  \
    if (nullptr == pointer)                 \
{                                       \
    IO_THROW_EXCEPTION(std::string(typeid(pointer).name()) + std::string(" ") + std::string(#pointer) + " is null.");                \
}
#endif

MED_IMAGING_END_NAMESPACE

#endif