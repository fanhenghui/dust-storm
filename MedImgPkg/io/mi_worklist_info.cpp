#include "mi_worklist_info.h"

#include <iostream>

MED_IMG_BEGIN_NAMESPACE

WorkListInfo::WorkListInfo()
    : m_sPatientID(""), m_sPatientName(""), m_sPatientSex(""),
      m_sPatientBirthday(""), m_sPatientAge(""), m_sPatientHeight(""),
      m_sPatientWeight(""), m_sAccessionID(""), m_sStudyInsUID(""),
      m_sReferringPhysician(""), m_sRequestingPhysician(""),
      m_sOperatorsName(""), m_sStudyDate(""), m_sStudyTime(""),
      m_sStudyDescription(""), m_sModality(""), m_sScheduledAETitle("") {}

WorkListInfo::~WorkListInfo() {}

const std::string& WorkListInfo::GetPatientID() const {
    return this->m_sPatientID;
}

const std::string& WorkListInfo::GetPatientName() const {
    return this->m_sPatientName;
}

const std::string& WorkListInfo::GetPatientSex() const {
    return this->m_sPatientSex;
}

const std::string& WorkListInfo::GetPatientAge() const {
    return this->m_sPatientAge;
}

const std::string& WorkListInfo::GetPatientHeight() const {
    return this->m_sPatientHeight;
}

const std::string& WorkListInfo::GetPatientWeight() const {
    return this->m_sPatientWeight;
}

const std::string& WorkListInfo::GetAccessionNo() const {
    return this->m_sAccessionID;
}

const std::string& WorkListInfo::GetStudyInsUID() const {
    return this->m_sStudyInsUID;
}

const std::string& WorkListInfo::GetSeriesInsUID() const {
    return this->m_sSeriesInsUID;
}

const std::string& WorkListInfo::GetReferringPhysician() const {
    return this->m_sReferringPhysician;
}

const std::string& WorkListInfo::GetRequestingPhysician() const {
    return this->m_sRequestingPhysician;
}

const std::string& WorkListInfo::GetOperatorsName() const {
    return this->m_sOperatorsName;
}

const std::string& WorkListInfo::GetStudyDate() const {
    return this->m_sStudyDate;
}

const std::string& WorkListInfo::GetStudyTime() const {
    return this->m_sStudyTime;
}

const std::string& WorkListInfo::GetStudyDescription() const {
    return this->m_sStudyDescription;
}

const std::string& WorkListInfo::GetModality() const {
    return this->m_sModality;
}

const std::string& WorkListInfo::GetScheduledAETitle() const {
    return this->m_sScheduledAETitle;
}

bool WorkListInfo::SetPatientID(const std::string& sPatientID) {
    this->m_sPatientID = sPatientID;
    return true;
}

bool WorkListInfo::SetPatientName(const std::string& sPatientName) {

    this->m_sPatientName = sPatientName;
    return true;
}

bool WorkListInfo::SetPatientSex(const std::string& sPatientSex) {
    if (sPatientSex != "F" && sPatientSex != "M" && sPatientSex != "O" &&
            sPatientSex != "f" && sPatientSex != "m" && sPatientSex != "o") {
        std::cout << "WorkListInfo::SetPatientSex() Parameter Check: PatientSex "
                  "isn't 'F', 'f','M', 'm' or 'O', 'o'."
                  << std::endl;
        return false;
    }

    if ("F" == sPatientSex || "f" == sPatientSex) {
        this->m_sPatientSex = "F";
    }

    if ("M" == sPatientSex || "m" == sPatientSex) {
        this->m_sPatientSex = "M";
    }

    if ("O" == sPatientSex || "o" == sPatientSex) {
        this->m_sPatientSex = "O";
    }

    return true;
}

bool WorkListInfo::SetPatientAge(const std::string& sPatientAge) {
    this->m_sPatientAge = sPatientAge;
    return true;
}

bool WorkListInfo::SetPatientHeight(const std::string& sPatientHeight) {
    this->m_sPatientHeight = sPatientHeight;
    return true;
}

bool WorkListInfo::SetPatientWeight(const std::string& sPatientWeight) {
    this->m_sPatientWeight = sPatientWeight;
    return true;
}

bool WorkListInfo::SetAccessionNo(const std::string& sAccessionNo) {
    this->m_sAccessionID = sAccessionNo;
    return true;
}

bool WorkListInfo::SetStudyInsUID(const std::string& sStudyInsUID) {
    this->m_sStudyInsUID = sStudyInsUID;
    return true;
}

bool WorkListInfo::SetSeriesInsUID(const std::string& sSeriesInsUID) {
    this->m_sSeriesInsUID = sSeriesInsUID;
    return true;
}

bool WorkListInfo::SetReferringPhysician(
    const std::string& sReferringPhysician) {
    this->m_sReferringPhysician = sReferringPhysician;
    return true;
}

bool WorkListInfo::SetRequestingPhysician(
    const std::string& sRequestingPhysician) {
    this->m_sRequestingPhysician = sRequestingPhysician;
    return true;
}

bool WorkListInfo::SetOperatorsName(const std::string& sOperatorsName) {
    this->m_sOperatorsName = sOperatorsName;
    return true;
}

bool WorkListInfo::SetStudyDate(const std::string& sStudyDate) {
    this->m_sStudyDate = sStudyDate;
    return true;
}

bool WorkListInfo::SetStudyTime(const std::string& sStudyTime) {
    this->m_sStudyTime = sStudyTime;
    return true;
}

bool WorkListInfo::SetStudyDescription(const std::string& sStudyDescription) {
    this->m_sStudyDescription = sStudyDescription;
    return true;
}

bool WorkListInfo::SetModality(const std::string& sModality) {
    this->m_sModality = sModality;
    return true;
}

bool WorkListInfo::SetScheduledAETitle(const std::string& sScheduledAETitle) {
    this->m_sScheduledAETitle = sScheduledAETitle;
    return true;
}

void WorkListInfo::SetPatientBirthday(const std::string& sPatientBirthday) {
    this->m_sPatientBirthday = sPatientBirthday;
}

const std::string& WorkListInfo::GetPaitentBirthday() const {
    return this->m_sPatientBirthday;
}

const std::string& WorkListInfo::GetRequestedProcedureID() const {
    return m_sRequestedProcedureID;
}

void WorkListInfo::SetRequestedProcedureID(
    const std::string& sRequestedProcedureID) {
    m_sRequestedProcedureID = sRequestedProcedureID;
}

const std::string& WorkListInfo::GetScheduledProcedureStepID() const {
    return m_sScheduledProcedureStepID;
}

void WorkListInfo::SetScheduledProcedureStepID(
    const std::string& sScheduledProcedureStepID) {
    m_sScheduledProcedureStepID = sScheduledProcedureStepID;
}

const std::string& WorkListInfo::GetScheduledProcedureStepDescription() const {
    return m_sScheduledProcedureStepDescription;
}

void WorkListInfo::SetScheduledProcedureStepDescription(
    const std::string& sScheduledProcedureStepDescription) {
    m_sScheduledProcedureStepDescription = sScheduledProcedureStepDescription;
}

const std::string& WorkListInfo::GetScheduledProcedureStepStartDate() const {
    return m_sScheduledProcedureStepStartDate;
}

void WorkListInfo::SetScheduledProcedureStepStartDate(
    const std::string& sScheduledProcedureStepStartDate) {
    m_sScheduledProcedureStepStartDate = sScheduledProcedureStepStartDate;
}

const std::string& WorkListInfo::GetScheduledProcedureStepStartTime() const {
    return m_sScheduledProcedureStepStartTime;
}

void WorkListInfo::SetScheduledProcedureStepStartTime(
    const std::string& sScheduledProcedureStepStartTime) {
    m_sScheduledProcedureStepStartTime = sScheduledProcedureStepStartTime;
}

const std::string& WorkListInfo::GetRequestedProcedureDescription() const {
    return m_sRequestedProcedureDescription;
}

void WorkListInfo::SetRequestedProcedureDescription(
    const std::string& sRequestedProcedureDescription) {
    m_sRequestedProcedureDescription = sRequestedProcedureDescription;
}

MED_IMG_END_NAMESPACE