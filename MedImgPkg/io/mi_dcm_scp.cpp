#include "mi_dcm_scp.h"

MED_IMG_BEGIN_NAMESPACE

MIDcmSCP::MIDcmSCP():_stop(false) {
}

MIDcmSCP::~MIDcmSCP() {
}

void MIDcmSCP::stop() {
    _stop = true;
}

OFCondition MIDcmSCP::handleIncomingCommand(T_DIMSE_Message* incomingMsg,
                                const DcmPresentationContextInfo& presInfo) {
    return this->DcmStorageSCP::handleIncomingCommand(incomingMsg, presInfo);
}

OFBool MIDcmSCP::stopAfterCurrentAssociation() {
    return _stop;
}

MED_IMG_END_NAMESPACE