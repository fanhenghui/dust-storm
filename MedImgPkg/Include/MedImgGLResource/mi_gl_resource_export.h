#ifndef MED_IMG_GL_RESOURCE_H
#define MED_IMG_GL_RESOURCE_H

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
    #ifdef MEDIMGGLRESOURCE_EXPORTS
    #define GLResource_Export __declspec(dllexport)
    #else
    #define GLResource_Export __declspec(dllimport)
    #endif
#else
    #define GLResource_Export
#endif

#ifndef GLRESOURCE_THROW_EXCEPTION
#define GLRESOURCE_THROW_EXCEPTION(desc) THROW_EXCEPTION("GLResource" , desc);
#endif

#ifndef GLRESOURCE_CHECK_NULL_EXCEPTION
#define  GLRESOURCE_CHECK_NULL_EXCEPTION(pointer)                  \
    if (nullptr == pointer)                 \
{                                       \
    GLRESOURCE_THROW_EXCEPTION(std::string(typeid(pointer).name()) + std::string(" ") + std::string(#pointer) + " is null.");                \
}
#endif

MED_IMG_END_NAMESPACE

#endif