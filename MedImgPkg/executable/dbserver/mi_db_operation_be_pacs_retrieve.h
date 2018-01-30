#ifndef MEDIMG_DB_MI_DB_OPERATION_BE_PACS_FETCH_H
#define MEDIMG_DB_MI_DB_OPERATION_BE_PACS_FETCH_H

#include "util/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpBEPACSRetrieve: public IOperation {
public:
    DBOpBEPACSRetrieve();
    virtual ~DBOpBEPACSRetrieve();

    virtual int execute();

    CREATE_EXTENDS_OP(DBOpBEPACSRetrieve)
private:
    int preprocess(const std::string& series_dir, const std::string& preprocess_mask_path, float& dicoms_size_mb);
};

MED_IMG_END_NAMESPACE

#endif