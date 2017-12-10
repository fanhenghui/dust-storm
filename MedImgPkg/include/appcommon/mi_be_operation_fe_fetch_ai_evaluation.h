#ifndef MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_FETCH_AI_EVALUATION_H
#define MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_FETCH_AI_EVALUATION_H

#include "appcommon/mi_app_common_export.h"
#include "util/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export BEOpFEFetchAIEvaluation : public IOperation {
public:
    BEOpFEFetchAIEvaluation();
    virtual ~BEOpFEFetchAIEvaluation();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<BEOpFEFetchAIEvaluation>(new BEOpFEFetchAIEvaluation());
    }
};
MED_IMG_END_NAMESPACE

#endif