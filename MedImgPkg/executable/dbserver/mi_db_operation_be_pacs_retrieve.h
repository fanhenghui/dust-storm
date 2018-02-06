#ifndef MEDIMG_DB_MI_DB_OPERATION_BE_PACS_FETCH_H
#define MEDIMG_DB_MI_DB_OPERATION_BE_PACS_FETCH_H

#include "util/mi_operation_interface.h"
#include "io/mi_dicom_info.h"

MED_IMG_BEGIN_NAMESPACE

class DBOpBEPACSRetrieve: public IOperation {
public:
    DBOpBEPACSRetrieve();
    virtual ~DBOpBEPACSRetrieve();

    virtual int execute();

    CREATE_EXTENDS_OP(DBOpBEPACSRetrieve)
private:
    int preprocess(const std::vector<InstanceInfo>& instances, const std::string& preprocess_mask_path);
};

MED_IMG_END_NAMESPACE

#endif