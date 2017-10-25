#ifndef MED_IMG_APPCOMMON_MI_OPERATION_ROTATE_H
#define MED_IMG_APPCOMMON_MI_OPERATION_ROTATE_H

#include "appcommon/mi_app_common_export.h"
#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export OpRotate : public IOperation {
public:
    OpRotate();
    virtual ~OpRotate();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<OpRotate>(new OpRotate());
    }

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif