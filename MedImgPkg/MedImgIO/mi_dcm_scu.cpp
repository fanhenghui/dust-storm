#include "mi_dcm_scu.h"

#include "mi_worklist_info.h"

#include "dcmtk/config/osconfig.h"  /* make sure OS specific configuration is included first */ 
#include "dcmtk/dcmnet/diutil.h" 

MED_IMG_BEGIN_NAMESPACE

static Uint8 findUncompressedPC(const OFString& sopClass, MIDcmSCU& scu)
{
    Uint8 pc;
    pc = scu.findPresentationContextID(sopClass, UID_LittleEndianExplicitTransferSyntax);
    if (pc == 0)
        pc = scu.findPresentationContextID(sopClass, UID_BigEndianExplicitTransferSyntax);
    if (pc == 0)
        pc = scu.findPresentationContextID(sopClass, UID_LittleEndianImplicitTransferSyntax);
    return pc;
}
MIDcmSCU::MIDcmSCU(const char * self_AE_title) : _association_ready(false), _p_work_list(nullptr)
{
    // set AE titles 
    this->setAETitle(self_AE_title);
}

bool MIDcmSCU::createAssociation(const char * serive_ip_address, const unsigned short serive_port, const char * service_AE_title)
{
    if (!this->_association_ready)
    {
        this->setPeerHostName(serive_ip_address);
        this->setPeerPort(serive_port);
        this->setPeerAETitle(service_AE_title);

        OFList<OFString> ts;
        ts.push_back(UID_LittleEndianExplicitTransferSyntax);
        ts.push_back(UID_BigEndianExplicitTransferSyntax);
        ts.push_back(UID_LittleEndianImplicitTransferSyntax);
        
        // TODO: list all the service we would like to use
        this->addPresentationContext(UID_VerificationSOPClass, ts);
        this->addPresentationContext(UID_FINDStudyRootQueryRetrieveInformationModel, ts);
        this->addPresentationContext(UID_MOVEStudyRootQueryRetrieveInformationModel, ts);

        /* Initialize network */
        OFCondition result = this->initNetwork();
        if (result.bad())
        {
            DCMNET_ERROR("Unable to set up the network: " << result.text());
            return this->_association_ready;
        }

        /* Negotiate Association */
        result = this->negotiateAssociation();
        if (result.bad())
        {
            DCMNET_ERROR("Unable to negotiate association: " << result.text());
            return this->_association_ready;
        }

        /* Let's look whether the server is listening:
        Assemble and send C-ECHO request
        */
        result = this->sendECHORequest(0);
        if (result.bad())
        {
            DCMNET_ERROR("Could not process C-ECHO with the server: " << result.text());
            return this->_association_ready;
        }

        this->_association_ready = true;
    }
    return this->_association_ready;
}

bool MIDcmSCU::search_all()
{
    DcmDataset queryKeys;
    queryKeys.putAndInsertString(DCM_QueryRetrieveLevel, "SERIES");
    queryKeys.putAndInsertString(DCM_StudyInstanceUID, "");
    queryKeys.putAndInsertString(DCM_SeriesInstanceUID, "");
    queryKeys.putAndInsertString(DCM_SeriesNumber, "");
    queryKeys.putAndInsertString(DCM_PatientName, "");
    queryKeys.putAndInsertString(DCM_PatientID, "");
    queryKeys.putAndInsertString(DCM_PatientSex, "");
    queryKeys.putAndInsertString(DCM_Modality, "");

    return this->search(queryKeys);
}

bool MIDcmSCU::search(DcmDataset & query_keys)
{
    T_ASC_PresentationContextID presID = findUncompressedPC(UID_FINDStudyRootQueryRetrieveInformationModel, *this);
    if (presID == 0)
    {
        DCMNET_ERROR("There is no uncompressed presentation context for Study Root FIND");
        return false;
    }
    OFCondition result = this->sendFINDRequest(presID, &query_keys, NULL /* we are not interested into responses*/);
    if (result.bad())
        return false;
    else
        //DCMNET_INFO("There are " << findResponses.size() << " studies available");
        return true;
}


bool MIDcmSCU::fetch(const char * dst_AE_title, const WorkListInfo& which_one)
{
    DcmDataset queryKeys;
    queryKeys.putAndInsertString(DCM_QueryRetrieveLevel, "SERIES");
    queryKeys.putAndInsertString(DCM_StudyInstanceUID, which_one.GetStudyInsUID().c_str());
    queryKeys.putAndInsertString(DCM_SeriesInstanceUID, which_one.GetSeriesInsUID().c_str());

    return this->fetch(dst_AE_title, queryKeys);
}


bool MIDcmSCU::fetch(const char * dst_AE_title, DcmDataset & query_keys)
{
    // fetches all images of this particular study
    T_ASC_PresentationContextID presID = findUncompressedPC(UID_MOVEStudyRootQueryRetrieveInformationModel, *this);
    if (presID == 0)
    {
        DCMNET_ERROR("There is no uncompressed presentation context for Study Root MOVE");
        return false;
    }
    
    // start an additional storage server  (e.g., storescp.exe) before calling sendMOVERequest

    OFCondition result = this->sendMOVERequest(presID, dst_AE_title, &query_keys, NULL /* we are not interested into responses*/);
    if (result.bad())
    {
        return false;
    }
    else
        return true;
}

void MIDcmSCU::endAssociation()
{
    this->releaseAssociation();
    this->freeNetwork();
    this->_association_ready = false;
}

OFCondition MIDcmSCU::handleFINDResponse(const T_ASC_PresentationContextID presID, QRResponse *response, OFBool &waitForNextResponse)
{
    if (this->_p_work_list != nullptr)
    {
        this->addFindResult2List(response, *this->_p_work_list);
    }

    // call base handler
    return DcmSCU::handleFINDResponse(presID, response, waitForNextResponse);
}

void MIDcmSCU::set_work_list(std::vector<WorkListInfo> * p_work_list)
{
    this->_p_work_list = p_work_list;
}

void MIDcmSCU::addFindResult2List(QRResponse *response, std::vector<WorkListInfo>& add_to_list)
{
    if (response->m_dataset != nullptr)
    {
        const char* cCharacterSet = NULL;
        const char* cPatientID = NULL;
        const char* cPatientName = NULL;
        const char* cPatientSex = NULL;
        const char* cPatientBirthday = NULL;
        const char* cStudyDate = NULL;
        const char* cStudyTime = NULL;
        const char* cAccessionNo = NULL;
        const char* cStudyInsUID = NULL;
        const char* cSeriesInsUID = NULL;
        const char* cStudyDescription = NULL;
        const char* cReferringPhysician = NULL;
        const char* cRequestingPhysician = NULL;
        const char* cOperatorsName = NULL;
        const char* cPatientAge = NULL;
        const char* cModality = NULL;
        const char* cScheduledAETitle = NULL;
        const char* cPatientHeight = NULL;
        const char* cPatientWeight = NULL;
        const char* cRequestedProcedureID = NULL;
        const char* cScheduledProStepID = NULL;
        const char* cScheduledProStepDes = NULL;
        const char* cScheduledProStepStartDate = NULL;
        const char* cScheduledProStepStartTime = NULL;
        const char* cRequestedProDes = NULL;
        
        DcmDataset * responseIdentifiers = response->m_dataset;
        responseIdentifiers->findAndGetString(DCM_SpecificCharacterSet, cCharacterSet);
        responseIdentifiers->findAndGetString(DCM_PatientID, cPatientID);
        responseIdentifiers->findAndGetString(DCM_PatientName, cPatientName);
        responseIdentifiers->findAndGetString(DCM_PatientSex, cPatientSex);
        responseIdentifiers->findAndGetString(DCM_PatientBirthDate, cPatientBirthday);
        responseIdentifiers->findAndGetString(DCM_StudyDate, cStudyDate);
        responseIdentifiers->findAndGetString(DCM_StudyTime, cStudyTime);
        responseIdentifiers->findAndGetString(DCM_AccessionNumber, cAccessionNo);
        responseIdentifiers->findAndGetString(DCM_StudyInstanceUID, cStudyInsUID);
        responseIdentifiers->findAndGetString(DCM_SeriesInstanceUID, cSeriesInsUID);
        responseIdentifiers->findAndGetString(DCM_StudyDescription, cStudyDescription);
        responseIdentifiers->findAndGetString(DCM_ReferringPhysicianName, cReferringPhysician);
        responseIdentifiers->findAndGetString(DCM_RequestingPhysician, cRequestingPhysician);
        responseIdentifiers->findAndGetString(DCM_OperatorsName, cOperatorsName);
        responseIdentifiers->findAndGetString(DCM_PatientAge, cPatientAge);
        responseIdentifiers->findAndGetString(DCM_Modality, cModality, true);
        responseIdentifiers->findAndGetString(DCM_ScheduledStationAETitle, cScheduledAETitle, true);
        responseIdentifiers->findAndGetString(DCM_PatientSize, cPatientHeight);
        responseIdentifiers->findAndGetString(DCM_PatientWeight, cPatientWeight);
        responseIdentifiers->findAndGetString(DCM_RequestedProcedureID, cRequestedProcedureID, true);
        responseIdentifiers->findAndGetString(DCM_ScheduledProcedureStepID, cScheduledProStepID, true);
        responseIdentifiers->findAndGetString(DCM_ScheduledProcedureStepDescription, cScheduledProStepDes, true);
        responseIdentifiers->findAndGetString(DCM_ScheduledProcedureStepStartDate, cScheduledProStepStartDate, true);
        responseIdentifiers->findAndGetString(DCM_ScheduledProcedureStepStartTime, cScheduledProStepStartTime, true);
        responseIdentifiers->findAndGetString(DCM_RequestedProcedureDescription, cRequestedProDes, true);

        WorkListInfo tempWorklistInfo;
        bool bRet = false;
        if (NULL != cPatientID)
        {
            std::string sPatientID(cPatientID);
            bRet = tempWorklistInfo.SetPatientID(sPatientID);
        }

        if (NULL != cPatientName)
        {
            std::string sPatientName(cPatientName);
            std::string sTemp = sPatientName;
            //ConvertToUtf8(cCharacterSet, sPatientName, sTemp);
            //if (Trim(sTemp)) {};
            bRet = tempWorklistInfo.SetPatientName(sTemp);
        }

        if (NULL != cPatientSex)
        {
            std::string sPatientSex(cPatientSex);
            sPatientSex = sPatientSex.front();
            bRet = tempWorklistInfo.SetPatientSex(sPatientSex);
        }

        if (NULL != cPatientBirthday)
        {
            std::string sPatientBirthday(cPatientBirthday);
            tempWorklistInfo.SetPatientBirthday(sPatientBirthday);
        }

        if (NULL != cPatientAge)
        {
            std::string sPatientAge(cPatientAge);
            //if (Trim(sPatientAge)) {};
            tempWorklistInfo.SetPatientAge(sPatientAge);
        }

        if (NULL != cPatientHeight)
        {
            std::string sPatientHeight(cPatientHeight);
            tempWorklistInfo.SetPatientHeight(sPatientHeight);
        }

        if (NULL != cPatientWeight)
        {
            std::string sPatientWeight(cPatientWeight);
            tempWorklistInfo.SetPatientWeight(sPatientWeight);
        }

        if (NULL != cStudyDate)
        {
            std::string sStudyDate(cStudyDate);
            bRet = tempWorklistInfo.SetStudyDate(sStudyDate);
        }

        if (NULL != cStudyTime)
        {
            std::string sStudyTime(cStudyTime);
            bRet = tempWorklistInfo.SetStudyTime(sStudyTime);
        }

        if (NULL != cStudyInsUID)
        {
            std::string sStudyInsUID(cStudyInsUID);
            bRet = tempWorklistInfo.SetStudyInsUID(sStudyInsUID);
        }

        if (NULL != cSeriesInsUID)
        {
            std::string sSeriesInsUID(cSeriesInsUID);
            bRet = tempWorklistInfo.SetSeriesInsUID(sSeriesInsUID);
        }

        if (NULL != cStudyDescription)
        {
            std::string sStudyDescription(cStudyDescription);
            std::string sTemp = sStudyDescription;
            //ConvertToUtf8(cCharacterSet, sStudyDescription, sTemp);
            //if (Trim(sTemp)) {};
            bRet = tempWorklistInfo.SetStudyDescription(sTemp);
        }

        if (NULL != cReferringPhysician)
        {
            std::string sReferringPhysician(cReferringPhysician);
            std::string sTemp = sReferringPhysician;
            //ConvertToUtf8(cCharacterSet, sReferringPhysician, sTemp);
            //if (Trim(sTemp)) {};
            bRet = tempWorklistInfo.SetReferringPhysician(sTemp);
        }

        if (NULL != cRequestingPhysician)
        {
            std::string sRequestingPhysician(cRequestingPhysician);
            std::string sTemp = sRequestingPhysician;
            //ConvertToUtf8(cCharacterSet, sRequestingPhysician, sTemp);
            //if (Trim(sTemp)) {};
            bRet = tempWorklistInfo.SetRequestingPhysician(sTemp);
        }

        if (NULL != cOperatorsName)
        {
            std::string sOperatorsName(cOperatorsName);
            std::string sTemp = sOperatorsName;
            //ConvertToUtf8(cCharacterSet, sOperatorsName, sTemp);
            //if (Trim(sTemp)) {};
            bRet = tempWorklistInfo.SetOperatorsName(sTemp);
        }

        if (NULL != cModality)
        {
            std::string sModality(cModality);
            bRet = tempWorklistInfo.SetModality(sModality);
        }

        if (NULL != cScheduledAETitle)
        {
            std::string sScheduledAETitle(cScheduledAETitle);
            bRet = tempWorklistInfo.SetScheduledAETitle(sScheduledAETitle);
        }

        if (NULL != cAccessionNo)
        {
            std::string sAccessionNo(cAccessionNo);
            bRet = tempWorklistInfo.SetAccessionNo(sAccessionNo);
        }

        if (NULL != cRequestedProcedureID)
        {
            std::string sRequestedProcedureID(cRequestedProcedureID);
            tempWorklistInfo.SetRequestedProcedureID(sRequestedProcedureID);
        }

        if (NULL != cScheduledProStepID)
        {
            std::string sScheduledProStepID(cScheduledProStepID);
            tempWorklistInfo.SetScheduledProcedureStepID(sScheduledProStepID);
        }

        if (NULL != cScheduledProStepDes)
        {
            std::string sScheduledProStepDes(cScheduledProStepDes);
            std::string sTemp = sScheduledProStepDes;
            //ConvertToUtf8(cCharacterSet, sScheduledProStepDes, sTemp);
            //if (Trim(sTemp)) {};
            tempWorklistInfo.SetScheduledProcedureStepDescription(sTemp);
        }

        if (NULL != cScheduledProStepStartDate)
        {
            std::string sScheduledProStepStartDate(cScheduledProStepStartDate);
            tempWorklistInfo.SetScheduledProcedureStepStartDate(sScheduledProStepStartDate);
        }

        if (NULL != cScheduledProStepStartTime)
        {
            std::string sScheduledProStepStartTime(cScheduledProStepStartTime);
            tempWorklistInfo.SetScheduledProcedureStepStartTime(sScheduledProStepStartTime);
        }

        if (NULL != cRequestedProDes)
        {
            std::string sRequestedProDes(cRequestedProDes);
            std::string sTemp = sRequestedProDes;
            //ConvertToUtf8(cCharacterSet, sRequestedProDes, sTemp);
            //if (Trim(sTemp)) {};
            tempWorklistInfo.SetRequestedProcedureDescription(sTemp);
        }

        add_to_list.push_back(tempWorklistInfo);
    }
}

MED_IMG_END_NAMESPACE