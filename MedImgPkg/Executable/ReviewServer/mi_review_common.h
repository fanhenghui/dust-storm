#ifndef MED_IMG_REVIEW_H_H_H
#define MED_IMG_REVIEW_H_H_H

#include "MedImgUtil/mi_exception.h"
#include "med_img_pkg_config.h"

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
#define COMMAND_ID_FE_SHUT_DOWN 120000
#define COMMAND_ID_FE_READY 120001
#define COMMAND_ID_FE_OPERATION 120002
#define COMMAND_ID_FE_MPR_PLAY 120003
#define COMMAND_ID_FE_VR_PLAY 120004

// BE to FE
#define COMMAND_ID_BE_SEND_IMAGE 270002
#define COMMAND_ID_BE_READY 270003

// BE to BE OPERATION ID
#define OPERATION_ID_BE_READY 310000

// FE to BE OPERATION ID
#define OPERATION_ID_MPR_PAGING 310001
#define OPERATION_ID_PAN 310002
#define OPERATION_ID_ZOOM 310003
#define OPERATION_ID_ROTATE 310004

MED_IMG_END_NAMESPACE

#endif