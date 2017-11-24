#ifndef MEDIMGAIS_MI_AI_SERVER_COMMON_H
#define MEDIMGAIS_MI_AI_SERVER_COMMON_H

#include "util/mi_exception.h"
#include "util/mi_ipc_common.h"
#include "appcommon/mi_app_common_define.h"
#include "med_img_pkg_config.h"

MED_IMG_BEGIN_NAMESPACE

#ifndef AISERVER_THROW_EXCEPTION
#define AISERVER_THROW_EXCEPTION(desc) THROW_EXCEPTION("AIS", desc);
#endif

#ifndef AISERVER_CHECK_NULL_EXCEPTION
#define AISERVER_CHECK_NULL_EXCEPTION(pointer)                                   \
  if (nullptr == pointer) {                                                    \
    AISERVER_THROW_EXCEPTION(std::string(typeid(pointer).name()) +               \
                           std::string(" ") + std::string(#pointer) +          \
                           " is null.");                                       \
  }
#endif

MED_IMG_END_NAMESPACE

#endif