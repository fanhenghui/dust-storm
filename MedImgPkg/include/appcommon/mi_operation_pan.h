#ifndef MED_IMG_APPCOMMON_MI_OPERATION_PAN_H
#define MED_IMG_APPCOMMON_MI_OPERATION_PAN_H

#include "appcommon/mi_app_common_export.h"
#include "appcommon/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export OpPan : public IOperation {
public:
    OpPan();
    virtual ~OpPan();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<OpPan>(new OpPan());
    }

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif