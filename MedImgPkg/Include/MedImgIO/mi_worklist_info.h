#ifndef MED_IMG_WORKLIST_INFO_H
#define MED_IMG_WORKLIST_INFO_H

#include <string>
#include "MedImgIO/mi_io_export.h"

MED_IMG_BEGIN_NAMESPACE

class IO_Export WorkListInfo
{
public:
    /////////////////////////////////////////////////////////////////
    ///  \brief constructor
    ///
    ///  \param[in]    None
    ///  \param[out]   None
    ///  \return       None
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    WorkListInfo();

    /////////////////////////////////////////////////////////////////
    ///  \brief deconstructor
    ///
    ///  \param[in]    None
    ///  \param[out]   None
    ///  \return       None
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    ~WorkListInfo();

    /////////////////////////////////////////////////////////////////
    ///  \brief setting up patient ID
    ///
    ///  \param[in]    const std::string& sPatientID 
    ///  \param[out]   None
    ///  \return       bool
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    bool SetPatientID(const std::string& sPatientID);


    /////////////////////////////////////////////////////////////////
    ///  \brief setting up patient Name
    ///
    ///  \param[in]    const std::string& sPatientName 
    ///  \param[out]   None
    ///  \return       bool
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    bool SetPatientName(const std::string& sPatientName);

    /////////////////////////////////////////////////////////////////
    ///  \brief setting up patient Sex
    ///
    ///  \param[in]    const std::string& sPatientSex
    ///  \param[out]   None
    ///  \return       bool
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    bool SetPatientSex(const std::string& sPatientSex);

    /////////////////////////////////////////////////////////////////
    ///  \brief setting up patient Age
    ///
    ///  \param[in]    const std::string& sPatientAge
    ///  \param[out]   None
    ///  \return       bool
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    bool SetPatientAge(const std::string& sPatientAge);

    /////////////////////////////////////////////////////////////////
    ///  \brief setting up patient Height
    ///
    ///  \param[in]    const std::string& sPatientHeight
    ///  \param[out]   None
    ///  \return       bool
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    bool SetPatientHeight(const std::string& sPatientHeight);

    /////////////////////////////////////////////////////////////////
    ///  \brief setting up patient Weight
    ///
    ///  \param[in]    const std::string& sPatientWeight
    ///  \param[out]   None
    ///  \return       bool
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    bool SetPatientWeight(const std::string& sPatientWeight);

    /////////////////////////////////////////////////////////////////
    ///  \brief setting up Accession Number
    ///
    ///  \param[in]    const std::string& sAccessionNo 
    ///  \param[out]   None
    ///  \return       bool
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    bool SetAccessionNo(const std::string& sAccessionNo);

    /////////////////////////////////////////////////////////////////
    ///  \brief setting up Study Instance UID
    ///
    ///  \param[in]    const std::string& sStudyInsUID 
    ///  \param[out]   None
    ///  \return       bool
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    bool SetStudyInsUID(const std::string& sStudyInsUID);

    bool SetSeriesInsUID(const std::string& sSeriesInsUID);

    /////////////////////////////////////////////////////////////////
    ///  \brief setting up Referring Physician
    ///
    ///  \param[in]    const std::string& sReferringPhysician 
    ///  \param[out]   None
    ///  \return       bool
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    bool SetReferringPhysician(const std::string& sReferringPhysician);

    /////////////////////////////////////////////////////////////////
    ///  \brief setting up Requesting Physician
    ///
    ///  \param[in]    const std::string& sRequestingPhysician
    ///  \param[out]   None
    ///  \return       bool
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    bool SetRequestingPhysician(const std::string& sRequestingPhysician);

    /////////////////////////////////////////////////////////////////
    ///  \brief setting up Operators Name
    ///
    ///  \param[in]    const std::string& sOperatorsName
    ///  \param[out]   None
    ///  \return       bool
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    bool SetOperatorsName(const std::string& sOperatorsName);

    /////////////////////////////////////////////////////////////////
    ///  \brief setting up Scheduled Study Date
    ///
    ///  \param[in]    const std::string& sStudyDate 
    ///  \param[out]   None
    ///  \return       bool
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    bool SetStudyDate(const std::string& sStudyDate);

    /////////////////////////////////////////////////////////////////
    ///  \brief setting up Scheduled Study Time
    ///
    ///  \param[in]    const std::string& sStudyTime 
    ///  \param[out]   None
    ///  \return       bool
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    bool SetStudyTime(const std::string& sStudyTime);

    /////////////////////////////////////////////////////////////////
    ///  \brief setting up Study Description
    ///
    ///  \param[in]    const std::string& sStudyDescription 
    ///  \param[out]   None
    ///  \return       bool
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    bool SetStudyDescription(const std::string& sStudyDescription);

    /////////////////////////////////////////////////////////////////
    ///  \brief setting up Modality Type
    ///
    ///  \param[in]    const std::string& sModality 
    ///  \param[out]   None
    ///  \return       bool
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    bool SetModality(const std::string& sModality);

    /////////////////////////////////////////////////////////////////
    ///  \brief setting up Scheduled AE title
    ///
    ///  \param[in]    const std::string& sScheduledAETitle 
    ///  \param[out]   None
    ///  \return       bool
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    bool SetScheduledAETitle(const std::string& sScheduledAETitle);

    /////////////////////////////////////////////////////////////////
    ///  \brief get Patient ID
    ///
    ///  \param[in]    None
    ///  \param[out]   None
    ///  \return       const std::string
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    const std::string& GetPatientID() const;

    /////////////////////////////////////////////////////////////////
    ///  \brief get Patient Name
    ///
    ///  \param[in]    None
    ///  \param[out]   None
    ///  \return       const std::string
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    const std::string& GetPatientName() const;

    /////////////////////////////////////////////////////////////////
    ///  \brief get Patient Sex
    ///
    ///  \param[in]    None
    ///  \param[out]   None
    ///  \return       const std::string
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    const std::string& GetPatientSex() const;

    /////////////////////////////////////////////////////////////////
    ///  \brief get Patient Age
    ///
    ///  \param[in]    None
    ///  \param[out]   None
    ///  \return       const std::string
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    const std::string& GetPatientAge() const;

    /////////////////////////////////////////////////////////////////
    ///  \brief get Patient Height
    ///
    ///  \param[in]    None
    ///  \param[out]   None
    ///  \return       const std::string
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    const std::string& GetPatientHeight() const;

    /////////////////////////////////////////////////////////////////
    ///  \brief get Patient Weight
    ///
    ///  \param[in]    None
    ///  \param[out]   None
    ///  \return       const std::string
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    const std::string& GetPatientWeight() const;

    /////////////////////////////////////////////////////////////////
    ///  \brief get AccessionNo.
    ///
    ///  \param[in]    None
    ///  \param[out]   None
    ///  \return       const std::string
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    const std::string& GetAccessionNo() const;

    /////////////////////////////////////////////////////////////////
    ///  \brief get Study Instance UID
    ///
    ///  \param[in]    None
    ///  \param[out]   None
    ///  \return       const std::string
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    const std::string& GetStudyInsUID() const;

    const std::string& GetSeriesInsUID() const;
    /////////////////////////////////////////////////////////////////
    ///  \brief get Referring Physician
    ///
    ///  \param[in]    None
    ///  \param[out]   None
    ///  \return       const std::string
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    const std::string& GetReferringPhysician() const;

    /////////////////////////////////////////////////////////////////
    ///  \brief get Requesting Physician
    ///
    ///  \param[in]    None
    ///  \param[out]   None
    ///  \return       const std::string
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    const std::string& GetRequestingPhysician() const;

    /////////////////////////////////////////////////////////////////
    ///  \brief get Operators Name
    ///
    ///  \param[in]    None
    ///  \param[out]   None
    ///  \return       const std::string
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    const std::string& GetOperatorsName() const;

    /////////////////////////////////////////////////////////////////
    ///  \brief get Study Date
    ///
    ///  \param[in]    None
    ///  \param[out]   None
    ///  \return       const std::string
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    const std::string& GetStudyDate() const;

    /////////////////////////////////////////////////////////////////
    ///  \brief get Study Time
    ///
    ///  \param[in]    None
    ///  \param[out]   None
    ///  \return       const std::string
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    const std::string& GetStudyTime() const;

    /////////////////////////////////////////////////////////////////
    ///  \brief get Study Description
    ///
    ///  \param[in]    None
    ///  \param[out]   None
    ///  \return       const std::string
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    const std::string& GetStudyDescription() const;

    /////////////////////////////////////////////////////////////////
    ///  \brief get Modality
    ///
    ///  \param[in]    None
    ///  \param[out]   None
    ///  \return       const std::string
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    const std::string& GetModality() const;

    /////////////////////////////////////////////////////////////////
    ///  \brief get Scheduled AE title
    ///
    ///  \param[in]    None
    ///  \param[out]   None
    ///  \return       const std::string
    ///  \pre \e  
    /////////////////////////////////////////////////////////////////
    const std::string& GetScheduledAETitle() const;

    const std::string& GetPaitentBirthday() const;

    void SetPatientBirthday(const std::string& sPatientBirthday);

    const std::string& GetRequestedProcedureID() const;
    void SetRequestedProcedureID(const std::string& sRequestedProcedureID);

    const std::string& GetScheduledProcedureStepID() const;
    void SetScheduledProcedureStepID(const std::string& sScheduledProcedureStepID);

    const std::string& GetScheduledProcedureStepDescription() const;
    void SetScheduledProcedureStepDescription(const std::string& sScheduledProcedureStepDescription);

    const std::string& GetScheduledProcedureStepStartDate() const;
    void SetScheduledProcedureStepStartDate(const std::string& sScheduledProcedureStepStartDate);

    const std::string& GetScheduledProcedureStepStartTime() const;
    void SetScheduledProcedureStepStartTime(const std::string& sScheduledProcedureStepStartTime);

    const std::string& GetRequestedProcedureDescription() const;
    void SetRequestedProcedureDescription(const std::string& sRequestedProcedureDescription);

private:
    /// \brief Patient ID
    std::string m_sPatientID;

    /// \brief Patient Name
    std::string m_sPatientName;

    /// \brief Patient Sex( value with: M,F,O)
    std::string m_sPatientSex;

    /// \brief Patient Birthday
    std::string m_sPatientBirthday;

    /// \brief Patient Age
    std::string m_sPatientAge;

    /// \brief Patient Height
    std::string m_sPatientHeight;

    /// \brief Patient Weight
    std::string m_sPatientWeight;

    /// \brief Accesssion Number
    std::string m_sAccessionID;

    /// \brief Study Instance UID
    std::string m_sStudyInsUID;

    /// \brief Study Instance UID
    std::string m_sSeriesInsUID;

    /// \brief Referring Physician
    std::string m_sReferringPhysician;

    /// \brief Requesting Physician
    std::string m_sRequestingPhysician;

    /// \brief Operators Name
    std::string m_sOperatorsName;

    /// \brief Scheduled Study Date
    std::string m_sStudyDate;

    /// \brief Scheduled Study Time
    std::string m_sStudyTime;

    /// \brief Study Description
    std::string m_sStudyDescription;

    /// \brief Modality type( value with: CT,MR,CR,DX,US,MG,RF,OT...)
    std::string m_sModality;

    std::string m_sScheduledAETitle;

    std::string m_sRequestedProcedureID;

    std::string m_sScheduledProcedureStepID;

    std::string m_sScheduledProcedureStepDescription;

    std::string m_sScheduledProcedureStepStartDate;

    std::string m_sScheduledProcedureStepStartTime;

    std::string m_sRequestedProcedureDescription;
};

MED_IMG_END_NAMESPACE

#endif