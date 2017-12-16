#ifndef MED_IMG_OPERATION_INIT_H
#define MED_IMG_OPERATION_INIT_H

#include "util/mi_operation_interface.h"
#include "mi_review_common.h"

MED_IMG_BEGIN_NAMESPACE

class AppController;
class MsgInit;
class OpInit : public IOperation {
public:
    OpInit();
    virtual ~OpInit();
    virtual int execute();

    CREATE_EXTENDS_OP(OpInit)

private:
    int init_data(std::shared_ptr<AppController> controller, const std::string& series_uid, bool& preprocessing_mask);
    int init_cell(std::shared_ptr<AppController> controller, MsgInit* msg_init, bool preprocessing_mask);
    int init_model(std::shared_ptr<AppController> controller, MsgInit* msg_init);

    int load_dcm_from_cache_db(std::shared_ptr<AppController> controller, const std::string& series_uid, const std::string& local_dcm_path);
    int query_from_remote_db(std::shared_ptr<AppController> controller, const std::string& series_uid, bool data_in_cache, bool& preprocessing_mask);
};

MED_IMG_END_NAMESPACE

#endif