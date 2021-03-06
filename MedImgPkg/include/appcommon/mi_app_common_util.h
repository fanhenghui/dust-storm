#ifndef MED_IMG_APPCOMMON_MI_APP_COMMON_UTIL_H
#define MED_IMG_APPCOMMON_MI_APP_COMMON_UTIL_H

#include "appcommon/mi_app_common_export.h"
#include "appcommon/mi_app_common_define.h"
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class AppController;
class ModelAnnotation;
class ModelCrosshair;
class ModelDBSStatus;
class ModelAnonymization;
class ModelUser;
class ModelPACSCache;
class AppCommon_Export AppCommonUtil {
public:
    static std::shared_ptr<ModelAnnotation> get_model_annotation(std::shared_ptr<AppController> controller);
    static std::shared_ptr<ModelCrosshair> get_model_crosshair(std::shared_ptr<AppController> controller);
    static std::shared_ptr<ModelDBSStatus> get_model_dbs_status(std::shared_ptr<AppController> controller);
    static std::shared_ptr<ModelAnonymization> get_model_anonymization(std::shared_ptr<AppController> controller);
    static std::shared_ptr<ModelUser> get_model_user(std::shared_ptr<AppController> controller);
    static std::shared_ptr<ModelPACSCache> get_model_pacs_cache(std::shared_ptr<AppController> controller);
};

MED_IMG_END_NAMESPACE

#endif