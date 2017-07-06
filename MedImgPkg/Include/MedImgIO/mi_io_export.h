#ifndef MED_IMG_IO_H
#define MED_IMG_IO_H

#include "med_img_pkg_config.h"

#include <exception>
#include <memory>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cassert>

#include "MedImgUtil/mi_exception.h"


MED_IMG_BEGIN_NAMESPACE

#ifdef WIN32
    #ifdef MEDIMGIO_EXPORTS
    #define IO_Export __declspec(dllexport)
    #else
    #define IO_Export __declspec(dllimport)
    #endif
#else
    #define IO_Export 
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

MED_IMG_END_NAMESPACE

#endif