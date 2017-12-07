#ifndef MEDIMGIO_DCM_SCU_H
#define MEDIMGIO_DCM_SCU_H

#include <string>
#include <vector>

#include "dcmtk/dcmnet/scu.h"
#include "io/mi_io_export.h"

MED_IMG_BEGIN_NAMESPACE

class IO_Export MIDcmSCU : public DcmSCU {
public:
    MIDcmSCU();
    virtual ~MIDcmSCU();
protected:
    virtual OFCondition handleFINDResponse(const T_ASC_PresentationContextID, QRResponse* , OFBool& );
};

MED_IMG_END_NAMESPACE

#endif
