#ifndef MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_ADJUST_EVALUATION_PROBABILITY_H
#define MEDIMG_APPCOMMON_MI_BE_OPERATION_FE_ADJUST_EVALUATION_PROBABILITY_H

#include "appcommon/mi_app_common_export.h"
#include "util/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export BEOpFEAdjustEvaluationProbability : public IOperation {
public:
    BEOpFEAdjustEvaluationProbability();
    virtual ~BEOpFEAdjustEvaluationProbability();

    virtual int execute();

    CREATE_EXTENDS_OP(BEOpFEAdjustEvaluationProbability)
};

MED_IMG_END_NAMESPACE

#endif