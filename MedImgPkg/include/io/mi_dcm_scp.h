#ifndef MEDIMGIO_DCM_SCP_H
#define MEDIMGIO_DCM_SCP_H

#include "dcmtk/dcmnet/dstorscp.h"
#include "io/mi_io_export.h"

MED_IMG_BEGIN_NAMESPACE

class IO_Export MIDcmSCP : public DcmStorageSCP {
public:
    MIDcmSCP();
    virtual ~MIDcmSCP();

    void stop();

protected:
    virtual OFCondition handleIncomingCommand(T_DIMSE_Message* incomingMsg, const DcmPresentationContextInfo& presInfo);
    virtual OFBool stopAfterCurrentAssociation();

private:
    bool _stop;
};

MED_IMG_END_NAMESPACE

#endif