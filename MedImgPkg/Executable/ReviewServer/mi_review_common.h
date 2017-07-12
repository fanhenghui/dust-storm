#ifndef MED_IMG_REVIEW_H_H_H
#define MED_IMG_REVIEW_H_H_H

#include "med_img_pkg_config.h"
#include "MedImgUtil/mi_exception.h"

MED_IMG_BEGIN_NAMESPACE


#ifndef REVIEW_THROW_EXCEPTION
#define REVIEW_THROW_EXCEPTION(desc) THROW_EXCEPTION("Review" , desc);
#endif

#ifndef REVIEW_CHECK_NULL_EXCEPTION
#define  REVIEW_CHECK_NULL_EXCEPTION(pointer)                  \
    if (nullptr == pointer)                 \
{                                       \
    REVIEW_THROW_EXCEPTION(std::string(typeid(pointer).name()) + std::string(" ") + std::string(#pointer) + " is null.");                \
}
#endif

//FE to BE
#define COMMAND_ID_FE_READY 120001
#define COMMAND_ID_FE_OPERATION 120002
#define COMMAND_ID_FE_SHUT_DOWN 121112
#define COMMAND_ID_FE_LOAD_SERIES 120003
#define COMMAND_ID_FE_MPR_PLAY 120004

//BE to FE
#define COMMAND_ID_BE_READY 270001
#define COMMAND_ID_BE_SEND_IMAGE 270002

//OPERATION ID
#define OPERATION_ID_MPR_PAGING 310001


MED_IMG_END_NAMESPACE



#endif