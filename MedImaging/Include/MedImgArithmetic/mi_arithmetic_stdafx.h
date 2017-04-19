#ifndef MED_IMAGING_ARITHMETIC_H
#define MED_IMAGING_ARITHMETIC_H

#include "med_imaging_config.h"
#include "MedImgCommon/mi_common_exception.h"

#include <exception>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>
#include <cmath>

MED_IMAGING_BEGIN_NAMESPACE

#ifdef MEDIMGARITHMETIC_EXPORTS
#define Arithmetic_Export __declspec(dllexport)
#else
#define Arithmetic_Export __declspec(dllimport)
#endif

#pragma warning(disable: 4251)

#ifndef ARITHMETIC_THROW_EXCEPTION
#define ARITHMETIC_THROW_EXCEPTION(desc) THROW_EXCEPTION("Arithmetic" , desc);
#endif

MED_IMAGING_END_NAMESPACE

#endif