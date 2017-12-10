#ifndef MEDIMG_DB_MI_DB_OPERATION_BE_PACS_FETCH_H
#define MEDIMG_DB_MI_DB_OPERATION_BE_PACS_FETCH_H

#include "util/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpBEPACSFetch: public IOperation {
public:
    DBOpBEPACSFetch();
    virtual ~DBOpBEPACSFetch();

    virtual int execute();

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<DBOpBEPACSFetch>(new DBOpBEPACSFetch());
    }
private:
    int preprocess_i(const std::string& series_dir, const std::string& preprocess_mask_path);
};

MED_IMG_END_NAMESPACE

#endif