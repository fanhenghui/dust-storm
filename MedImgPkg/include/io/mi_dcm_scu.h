#ifndef MEDIMGIO_DCM_SCU_H
#define MEDIMGIO_DCM_SCU_H

#include "dcmtk/dcmnet/scu.h"
#include "io/mi_io_export.h"
#include <string>
#include <vector>

MED_IMG_BEGIN_NAMESPACE

class WorkListInfo;

class IO_Export MIDcmSCU : public DcmSCU {
public:
    MIDcmSCU(const char* self_AE_title);

    ~MIDcmSCU() {};

    // create association
    bool createAssociation(const char* serive_ip_address,
                           const unsigned short serive_port,
                           const char* service_AE_title);

    // search with user specified keys
    bool search_all();

    bool fetch(const char* dst_AE_title, const WorkListInfo& which_one);

    // finish the current association, without de-constructing, for next new
    // association
    void endAssociation();

    // output work list
    void set_work_list(std::vector<WorkListInfo>* p_work_list);

private:
    bool search(DcmDataset& query_keys);
    bool fetch(const char* dst_AE_title, DcmDataset& query_keys);

    // override virtual implementation
    virtual OFCondition
    handleFINDResponse(const T_ASC_PresentationContextID presID,
                       QRResponse* response, OFBool& waitForNextResponse);
    void addFindResult2List(QRResponse* response,
                            std::vector<WorkListInfo>& add_to_list);

private:
    bool _association_ready;
    std::vector<WorkListInfo>* _p_work_list;
};

MED_IMG_END_NAMESPACE

#endif
