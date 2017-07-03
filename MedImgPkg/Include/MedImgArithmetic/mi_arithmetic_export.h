#ifndef MED_IMG_ARITHMETIC_H
#define MED_IMG_ARITHMETIC_H

#include "med_img_pkg_config.h"
#include "MedImgUtil/mi_exception.h"

#include <exception>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>
#include <cmath>

MED_IMG_BEGIN_NAMESPACE

#ifdef WIN32
    #ifdef MEDIMGARITHMETIC_EXPORTS
    #define Arithmetic_Export __declspec(dllexport)
    #else
    #define Arithmetic_Export __declspec(dllimport)
    #endif
#else
    #define Arithmetic_Export 
#endif

#ifndef ARITHMETIC_THROW_EXCEPTION
#define ARITHMETIC_THROW_EXCEPTION(desc) THROW_EXCEPTION("Arithmetic" , desc);
#endif

#ifndef ARITHMETIC_CHECK_NULL_EXCEPTION
#define  ARITHMETIC_CHECK_NULL_EXCEPTION(pointer)                  \
    if (nullptr == pointer)                 \
{                                       \
    ARITHMETIC_THROW_EXCEPTION(std::string(typeid(pointer).name()) + std::string(" ") + std::string(#pointer) + " is null.");                \
}
#endif

MED_IMG_END_NAMESPACE

#endif