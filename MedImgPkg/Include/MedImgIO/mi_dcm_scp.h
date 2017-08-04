#ifndef MIA_DCM_SCP_H
#define MIA_DCM_SCP_H

#include "MedImgIO/mi_io_export.h"
#include "dcmtk/dcmnet/dstorscp.h"

MED_IMG_BEGIN_NAMESPACE

class IO_Export MIDcmSCP : public DcmStorageSCP
{
public:
    MIDcmSCP(const char * self_AE_title);
    ~MIDcmSCP() {};
    void initialize(const unsigned short port);

private:
    virtual OFCondition handleIncomingCommand(T_DIMSE_Message *incomingMsg, const DcmPresentationContextInfo &presInfo);
    void start_listening(const unsigned short port);
};

MED_IMG_END_NAMESPACE

#endif