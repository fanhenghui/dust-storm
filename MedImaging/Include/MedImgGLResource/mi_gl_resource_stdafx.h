#ifndef MED_IMAGING_GL_RESOURCE_H
#define MED_IMAGING_GL_RESOURCE_H

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

#ifdef MEDIMGGLRESOURCE_EXPORTS
#define GLResource_Export __declspec(dllexport)
#else
#define GLResource_Export __declspec(dllimport)
#endif

#pragma warning(disable: 4251)

#ifndef GLRESOURCE_THROW_EXCEPTION
#define GLRESOURCE_THROW_EXCEPTION(desc) THROW_EXCEPTION("GLResource" , desc);
#endif

MED_IMAGING_END_NAMESPACE

#endif