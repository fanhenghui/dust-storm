#ifndef MED_IMG_OPERATION_INIT_H
#define MED_IMG_OPERATION_INIT_H

#include <vector>
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
    int init_data(std::shared_ptr<AppController> controller, const std::string& series_uid, MsgInit* msg_init, bool& preprocessing_mask);
    int init_cell(std::shared_ptr<AppController> controller, const std::string& series_uid, MsgInit* msg_init, bool preprocessing_mask);
    int init_model(std::shared_ptr<AppController> controller);

    int load_dcm_from_cache_db(std::shared_ptr<AppController> controller, std::vector<std::string>& instance_file_paths);
    int query_from_remote_db(std::shared_ptr<AppController> controller, const std::string& series_uid, MsgInit* msg_init, bool dicom_in_cache, bool& preprocessing_mask);
};

MED_IMG_END_NAMESPACE

#endif