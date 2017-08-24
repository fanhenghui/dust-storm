#ifndef MED_IMG_REVIEW_H_H_H
#define MED_IMG_REVIEW_H_H_H

#include "MedImgUtil/mi_exception.h"
#include "med_img_pkg_config.h"

#include "MedImgAppCommon/mi_app_common_define.h"

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
#define COMMAND_ID_FE_MPR_PLAY 120003
#define COMMAND_ID_FE_VR_PLAY 120004

// FE to BE OPERATION ID
#define OPERATION_ID_INIT 310000
#define OPERATION_ID_MPR_PAGING 310001
#define OPERATION_ID_PAN 310002
#define OPERATION_ID_ZOOM 310003
#define OPERATION_ID_ROTATE 310004
#define OPERATION_ID_WINDOWING 310005
#define OPERATION_ID_RESIZE 310006

// BE to FE

MED_IMG_END_NAMESPACE

#endif