#ifndef MEDIMGRENDERALGO_RENDER_ALGORITHM_H
#define MEDIMGRENDERALGO_RENDER_ALGORITHM_H

#include "med_img_pkg_config.h"

#include "util/mi_exception.h"

MED_IMG_BEGIN_NAMESPACE

#ifdef WIN32
#ifdef MEDIMGRENDERALGORITHM_EXPORTS
#define RenderAlgo_Export __declspec(dllexport)
#else
#define RenderAlgo_Export __declspec(dllimport)
#endif
#else
#define RenderAlgo_Export
#endif

#ifndef RENDERALGO_THROW_EXCEPTION
#define RENDERALGO_THROW_EXCEPTION(desc)                                       \
  THROW_EXCEPTION("RenderAlgorithm", desc);
#endif

#ifndef RENDERALGO_CHECK_NULL_EXCEPTION
#define RENDERALGO_CHECK_NULL_EXCEPTION(pointer)                               \
  if (nullptr == pointer) {                                                    \
    RENDERALGO_THROW_EXCEPTION(std::string(typeid(pointer).name()) +           \
                               std::string(" ") + std::string(#pointer) +      \
                               " is null.");                                   \
  }
#endif

MED_IMG_END_NAMESPACE

#endif