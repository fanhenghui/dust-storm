#ifndef MED_IMG_REVIEW_H_H_H
#define MED_IMG_REVIEW_H_H_H

#include "util/mi_exception.h"
#include "med_img_pkg_config.h"

#include "appcommon/mi_app_common_define.h"

MED_IMG_BEGIN_NAMESPACE

#ifndef REVIEW_THROW_EXCEPTION
#define REVIEW_THROW_EXCEPTION(desc) THROW_EXCEPTION("Review", desc);
#endif

#ifndef REVIEW_CHECK_NULL_EXCEPTION
#define REVIEW_CHECK_NULL_EXCEPTION(pointer)                                   \
  if (nullptr == pointer) {                                                    \
    REVIEW_THROW_EXCEPTION(std::string(typeid(pointer).name()) +               \
                           std::string(" ") + std::string(#pointer) +          \
                           " is null.");                                       \
  }
#endif

// FE to BE
#define COMMAND_ID_WORKLIST 120005
// FE to BE OPERATION ID
#define OPERATION_ID_INIT 310000
#define OPERATION_ID_ANNOTATION 310007
// BE to FE

// Model ID
#define MI_MODEL_ID_ANNOTATION 410000

MED_IMG_END_NAMESPACE

#endif