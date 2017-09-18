#ifndef MEDIMGIO_DCM_SCP_H
#define MEDIMGIO_DCM_SCP_H

#include "dcmtk/dcmnet/dstorscp.h"
#include "io/mi_io_export.h"

MED_IMG_BEGIN_NAMESPACE

class MIDcmSCP : public DcmStorageSCP {
public:
    IO_Export MIDcmSCP(const char* self_AE_title);
    IO_Export ~MIDcmSCP() {};
    IO_Export void initialize(const unsigned short port);

private:
    virtual OFCondition
    handleIncomingCommand(T_DIMSE_Message* incomingMsg,
                          const DcmPresentationContextInfo& presInfo);
    void start_listening(const unsigned short port);
};

MED_IMG_END_NAMESPACE

#endif