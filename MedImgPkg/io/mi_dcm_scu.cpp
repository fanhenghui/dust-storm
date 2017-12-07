#include "mi_dcm_scu.h"

MED_IMG_BEGIN_NAMESPACE

MIDcmSCU::MIDcmSCU() {}

MIDcmSCU::~MIDcmSCU() {}

OFCondition MIDcmSCU::handleFINDResponse(const T_ASC_PresentationContextID id, QRResponse* res, OFBool& wait_flag) {
    return DcmSCU::handleFINDResponse(id, res, wait_flag);
}

MED_IMG_END_NAMESPACE