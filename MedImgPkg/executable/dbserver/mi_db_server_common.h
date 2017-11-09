#ifndef MED_IMG_DB_SERVER_COMMON_H
#define MED_IMG_DB_SERVER_COMMON_H

#include "util/mi_exception.h"
#include "med_img_pkg_config.h"

MED_IMG_BEGIN_NAMESPACE

#ifndef DBSERVER_THROW_EXCEPTION
#define DBSERVER_THROW_EXCEPTION(desc) THROW_EXCEPTION("Review", desc);
#endif

#ifndef DBSERVER_CHECK_NULL_EXCEPTION
#define DBSERVER_CHECK_NULL_EXCEPTION(pointer)                                   \
  if (nullptr == pointer) {                                                    \
    DBSERVER_THROW_EXCEPTION(std::string(typeid(pointer).name()) +               \
                           std::string(" ") + std::string(#pointer) +          \
                           " is null.");                                       \
  }
#endif

MED_IMG_END_NAMESPACE

#endif