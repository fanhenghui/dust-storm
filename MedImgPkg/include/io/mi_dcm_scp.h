#ifndef MEDIMGIO_DCM_SCP_H
#define MEDIMGIO_DCM_SCP_H

#include "dcmtk/dcmnet/dstorscp.h"
#include "io/mi_io_export.h"
#include "io/mi_dicom_info.h"
#include <vector>

MED_IMG_BEGIN_NAMESPACE

class IO_Export MIDcmSCP : public DcmStorageSCP {
public:
    MIDcmSCP();
    virtual ~MIDcmSCP();

    void stop();

    void set_instance_infos(std::vector<InstanceInfo>* instance_infos);

protected:
    virtual OFCondition handleIncomingCommand(T_DIMSE_Message* incomingMsg, const DcmPresentationContextInfo& presInfo);
    virtual OFBool stopAfterCurrentAssociation();

    virtual Uint16 checkAndProcessSTORERequest(const T_DIMSE_C_StoreRQ &reqMessage,
                                               DcmFileFormat &fileformat);
private:
    bool _stop;
    std::vector<InstanceInfo>* _instance_infos;
};

MED_IMG_END_NAMESPACE

#endif