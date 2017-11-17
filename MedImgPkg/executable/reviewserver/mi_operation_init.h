#ifndef MED_IMG_OPERATION_INIT_H
#define MED_IMG_OPERATION_INIT_H

#include "appcommon/mi_operation_interface.h"
#include "mi_review_common.h"

MED_IMG_BEGIN_NAMESPACE

class AppController;
class MsgInit;
class OpInit : public IOperation {
public:
    OpInit();
    virtual ~OpInit();
    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<OpInit>(new OpInit());
    }

private:
    int init_data_i(std::shared_ptr<AppController> controller, MsgInit* msg_init, bool& preprocessing_mask);
    int init_cell_i(std::shared_ptr<AppController> controller, MsgInit* msg_init, bool preprocessing_mask);
    int init_model_i(std::shared_ptr<AppController> controller, MsgInit* msg_init);

    int load_dcm_from_cache_db(std::shared_ptr<AppController> controller, MsgInit* msg_init, const std::string& local_dcm_path);
    int query_from_remote_db(std::shared_ptr<AppController> controller, MsgInit* msg_init, bool data_in_cache, bool& preprocessing_mask);
};

MED_IMG_END_NAMESPACE

#endif