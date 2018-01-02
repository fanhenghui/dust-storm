#ifndef MED_IMG_CUDARESOUECE_MI_CUDA_RESOURCE_EXPORT_H
#define MED_IMG_CUDARESOUECE_MI_CUDA_RESOURCE_EXPORT_H

#include "med_img_pkg_config.h"
#include "util/mi_exception.h"

MED_IMG_BEGIN_NAMESPACE

#ifdef WIN32
#ifdef MEDIMGCUDARESOURCE_EXPORTS
#define CUDAResource_Export __declspec(dllexport)
#else
#define CUDAResource_Export __declspec(dllimport)
#endif
#else
#define CUDAResource_Export
#endif

#ifndef CUDARESOURCE_THROW_EXCEPTION
#define CUDARESOURCE_THROW_EXCEPTION(desc) THROW_EXCEPTION("CUDAResource" , desc);
#endif

#ifndef CUDARESOURCE_CHECK_NULL_EXCEPTION
#define  CUDARESOURCE_CHECK_NULL_EXCEPTION(pointer)                  \
    if (nullptr == pointer)                 \
{                                       \
    CUDARESOURCE_THROW_EXCEPTION(std::string(typeid(pointer).name()) + std::string(" ") + std::string(#pointer) + " is null.");                \
}
#endif

MED_IMG_END_NAMESPACE

#endif