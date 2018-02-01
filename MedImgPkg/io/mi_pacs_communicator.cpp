#include "mi_pacs_communicator.h"
#include "mi_dcm_scu.h"
#include "mi_dcm_scp.h"
#include "mi_io_logger.h"

MED_IMG_BEGIN_NAMESPACE

//官方demo只给了没有压缩的三种传输类型，实际中应该如何处理？
inline Uint8 findUncompressedPC(const OFString& sopClass, DcmSCU& scu) { 
    Uint8 pc; 
    pc = scu.findPresentationContextID(sopClass, UID_LittleEndianExplicitTransferSyntax); 
    if (pc == 0) {
      pc = scu.findPresentationContextID(sopClass, UID_BigEndianExplicitTransferSyntax); 
    }
    if (pc == 0) { 
      pc = scu.findPresentationContextID(sopClass, UID_LittleEndianImplicitTransferSyntax); 
    }
    return pc; 
} 

const static std::string QUERY_LEVEL_PATIENT = "PATIENT";
const static std::string QUERY_LEVEL_STUDY = "STUDY";
const static std::string QUERY_LEVEL_SERIES = "SERIES";

inline void fill_dcm_info(DcmDataset* dataset, DcmInfo& info, const std::string& query_level) {
    OFString str;
    if (dataset->findAndGetOFString(DCM_StudyDate, str).good()) {
        info.study_date = std::string(str.c_str());
    }
    if (dataset->findAndGetOFString(DCM_StudyTime, str).good()) {
        info.study_time = std::string(str.c_str());
    }
    if (dataset->findAndGetOFString(DCM_Modality, str).good()) {
        info.modality = std::string(str.c_str());
    }
    if (dataset->findAndGetOFString(DCM_PatientID, str).good()) {
        info.patient_id = std::string(str.c_str());
    }
    if (dataset->findAndGetOFString(DCM_PatientName, str).good()) {
        info.patient_name = std::string(str.c_str());
    }
    if (dataset->findAndGetOFString(DCM_PatientSex, str).good()) {
        info.patient_sex = std::string(str.c_str());
    }
    if (dataset->findAndGetOFString(DCM_PatientBirthDate, str).good()) {
        info.patient_birth_date = std::string(str.c_str());
    }
    if (dataset->findAndGetOFString(DCM_AccessionNumber, str).good()) {
        info.accession_no = std::string(str.c_str());
    }
    if (dataset->findAndGetOFString(DCM_StudyInstanceUID, str).good()) {
        info.study_id = std::string(str.c_str());
    }
    if (dataset->findAndGetOFString(DCM_StudyDescription, str).good()) {
        info.study_desc = std::string(str.c_str());
    }
    if (dataset->findAndGetOFString(DCM_SeriesDescription, str).good()) {
        info.series_desc = std::string(str.c_str());
    }
    if (dataset->findAndGetOFString(DCM_InstitutionName, str).good()) {
        info.institution = std::string(str.c_str());
    }
    if (dataset->findAndGetOFString(DCM_SeriesNumber, str).good()) {
        info.series_no = std::string(str.c_str());
    }

    if (dataset->findAndGetOFString(DCM_SeriesInstanceUID, str).good()) {
        info.series_id = std::string(str.c_str());
    }
    if (dataset->findAndGetOFString(DCM_NumberOfStudyRelatedSeries, str).good()) {
        info.number_of_series = atoi(str.c_str());
    }
    if (QUERY_LEVEL_STUDY == query_level) {
        if (dataset->findAndGetOFString(DCM_NumberOfStudyRelatedInstances, str).good()) {
            info.number_of_instance = atoi(str.c_str());
        }
    } else if (QUERY_LEVEL_SERIES == query_level){
        if (dataset->findAndGetOFString(DCM_NumberOfSeriesRelatedInstances, str).good()) {
            info.number_of_instance = atoi(str.c_str());
        }
    }
}

struct PACSCommunicator::ConnectionCache  {
    std::string server_ae_title;
    std::string server_host;
    unsigned short server_port;
    std::string client_ae_title;
    unsigned short client_port;
};

PACSCommunicator::PACSCommunicator(bool open_dcmtk_console_log) {
    //Default close 
    if (open_dcmtk_console_log) {
        OFLog::configure(OFLogger::DEBUG_LOG_LEVEL);
    } else {
        OFLog::configure(OFLogger::OFF_LOG_LEVEL);
    }
}

PACSCommunicator::~PACSCommunicator() {
}

int PACSCommunicator::connect(const std::string& server_ae_title,const std::string& server_host, unsigned short server_port,
const std::string& client_ae_title, unsigned short client_port) {
    _connection_cache.reset(new ConnectionCache());
    _connection_cache->server_ae_title = server_ae_title;
    _connection_cache->server_host = server_host;
    _connection_cache->server_port = server_port;
    _connection_cache->client_ae_title = client_ae_title;
    _connection_cache->client_port = client_port;

    _scp_thread = boost::thread(boost::bind(&PACSCommunicator::run_scp, this));
    return try_connect();
}

void PACSCommunicator::disconnect() {
    if (_scp && _scu) {
        _scp->stop();

        DcmDataset query_key;
        query_key.putAndInsertString(DCM_QueryRetrieveLevel, "SERIES");
        query_key.putAndInsertString(DCM_SeriesInstanceUID, "xxx111x2x2x2x2x2");

        const T_ASC_PresentationContextID id = findUncompressedPC(UID_MOVEStudyRootQueryRetrieveInformationModel, *_scu);
        if (id == 0) {
            MI_IO_LOG(MI_ERROR) << "There is no uncompressed presentation context for Study Root MOVE";
        } else {
            //send last message to call SCP's stopAfterCurrentAssociation to return true , then stop listen
            //TODO 这里是有风险的：取了最近一次query all series 的第一个series，万一move失败，就不会走到stopAfterCurrentAssociation函数，因此没法关闭SCP
            MI_IO_LOG(MI_INFO) << "plase wait last query message.";
            _scu->sendMOVERequest(id, _connection_cache->client_ae_title.c_str(), &query_key, NULL);
            _scp_thread.join();        
        }
    }
    if (_scu) {
        _scu->closeAssociation(DCMSCU_RELEASE_ASSOCIATION);
    }

    MI_IO_LOG(MI_INFO) << "PACS communicator disconnect.";
}

int PACSCommunicator::try_connect() {
    if (!_connection_cache) {
        MI_IO_LOG(MI_FATAL) << "connection cache is null when try connect.";
        return -1;
    }

    //check old connection
    if (_scu) {
        OFCondition result = _scu->sendECHORequest(0);
        if (result.bad()) {
            MI_IO_LOG(MI_INFO) << "disconnection with remote. try create new connection.";
            _scu->closeAssociation(DCMSCU_RELEASE_ASSOCIATION);
        } else {
            MI_IO_LOG(MI_DEBUG) << "connection status good.";
            return 0;
        }
    }

    _scu.reset(new MIDcmSCU());
    _scu->setAETitle(_connection_cache->client_ae_title.c_str());
    _scu->setPeerAETitle(_connection_cache->server_ae_title.c_str());
    _scu->setPeerHostName(_connection_cache->server_host.c_str());
    _scu->setPeerPort(_connection_cache->server_port);

    // TODO 这里值列了非压缩的传输类型，如果需要列全的话，首先底层的DICOMLoader需要支持全
    OFList<OFString> ts;
    ts.push_back(UID_LittleEndianExplicitTransferSyntax);
    ts.push_back(UID_BigEndianExplicitTransferSyntax);
    ts.push_back(UID_LittleEndianImplicitTransferSyntax);
    _scu->addPresentationContext(UID_VerificationSOPClass, ts);
    _scu->addPresentationContext(UID_FINDStudyRootQueryRetrieveInformationModel, ts);
    _scu->addPresentationContext(UID_MOVEStudyRootQueryRetrieveInformationModel, ts);

    OFCondition result = _scu->initNetwork();
    if (result.bad()) {
        MI_IO_LOG(MI_FATAL) << "SCU initialize network failed.";
        return -1;
    }

    //Default connection timeout is -1(block until connected/refused)
    //Here set non-blocking mode to make progress going
    //DBS communicate with PACS will try connect before DIMSE 
    const int connection_timeout = _scu->getConnectionTimeout();
    //MI_IO_LOG(MI_DEBUG) << "SCU old connection timeout is " << connection_timeout << ".";//
    _scu->setConnectionTimeout(5);
    MI_IO_LOG(MI_INFO) << "SCU negotiating association in 5s.";
    result = _scu->negotiateAssociation();
    _scu->setConnectionTimeout(connection_timeout);

    if (result.bad()) {
        MI_IO_LOG(MI_FATAL) << "SCU negotiate association failed.";
        return -1;
    }
    MI_IO_LOG(MI_INFO) << "SCU negotiating association success.";

    result = _scu->sendECHORequest(0);
    if (result.bad()) {
        MI_IO_LOG(MI_FATAL) << "SCU echo server failed.";
        return -1;
    }

    MI_IO_LOG(MI_INFO) << "Connect to PACS {AET: " << _connection_cache->server_ae_title << "; URL: "
    << _connection_cache->server_host << ":" << _connection_cache->server_port << "} success.";

    return 0;
}

inline bool check_num(char num) {
    return (num >= '0' && num <= '9');
}

inline int date_valid(const std::string& date) {
    if (!date.empty()) {
        if (8 == date.size()) {
            for (auto it=date.begin(); it != date.end(); ++it) {
                if (!check_num(*it)) {
                    MI_IO_LOG(MI_ERROR) << "invalid date. 0";    
                    return -1;
                }
            }
        } else if (17 == date.size()) {
            for (int i=0; i<8; ++i) {
                if (!check_num(date[i])) {
                    MI_IO_LOG(MI_ERROR) << "invalid date. 1";    
                    return -1;
                }
            }
            if (date[8] != '-') {
                MI_IO_LOG(MI_ERROR) << "invalid date. 2";
                return -1;
            }
            for (int i=9; i<17; ++i) {
                if (!check_num(date[i])) {
                    MI_IO_LOG(MI_ERROR) << "invalid date. 3";    
                    return -1;
                }
            }
        } else {
            MI_IO_LOG(MI_ERROR) << "invalid date. 4";
            return -1;
        }   
    }

    return 0;
}

int PACSCommunicator::query(std::vector<DcmInfo>& dcm_infos, const QueryKey& key, QueryLevel elevel) {
    if(0 != try_connect() ) {
        MI_IO_LOG(MI_FATAL) << "try connect failed.";
        return -1;
    }

    std::string query_level = QUERY_LEVEL_PATIENT;
    if (elevel == STUDY) {
        query_level = QUERY_LEVEL_STUDY;
    } else if (elevel == SERIES) {
        query_level = QUERY_LEVEL_SERIES;
    }    

    DcmDataset query_key;
    query_key.putAndInsertString(DCM_QueryRetrieveLevel, query_level.c_str());

    static const int KEY_NUM = 10;
    static const DcmTag KEY_TAGS[KEY_NUM] = {
        DCM_StudyInstanceUID,
        DCM_SeriesInstanceUID,
        DCM_StudyDate,
        DCM_StudyTime,
        DCM_PatientID,
        DCM_PatientName,
        DCM_Modality,
        DCM_AccessionNumber,
        DCM_PatientSex,
        DCM_PatientBirthDate,
    };

    //check query valid
    if(-1 == date_valid(key.study_date)) {
        MI_IO_LOG(MI_ERROR) << "invalid query key: study date.";
        return -1;
    } 

    if(-1 == date_valid(key.patient_birth_date)) {
        MI_IO_LOG(MI_ERROR) << "invalid query key: patient birth date.";
        return -1;
    }

    if (!key.patient_sex.empty()) {
        if (!(key.patient_sex == "M" || key.patient_sex == "F" || key.patient_sex == "*")) {
            MI_IO_LOG(MI_ERROR) << "invalid query key: patient sex.";
            return -1;
        }
    }

    const std::string* KEYS[KEY_NUM] = {
        &key.study_uid,
        &key.series_uid,
        &key.study_date,
        &key.study_time,
        &key.patient_id,
        &key.patient_name,
        &key.modality,
        &key.accession_no,
        &key.patient_sex,
        &key.patient_birth_date,
    };
    for (int i=0; i<KEY_NUM; ++i) {
        query_key.putAndInsertString(KEY_TAGS[i], KEYS[i]->c_str());
    }

    query_key.putAndInsertString(DCM_StudyDescription, "");
    query_key.putAndInsertString(DCM_SeriesDescription, "");
    query_key.putAndInsertString(DCM_InstitutionName, "");
    query_key.putAndInsertString(DCM_SeriesNumber, "");
    query_key.putAndInsertString(DCM_NumberOfSeriesRelatedInstances, "");
    query_key.putAndInsertString(DCM_NumberOfStudyRelatedInstances, "");
    query_key.putAndInsertString(DCM_NumberOfStudyRelatedSeries, "");

    const T_ASC_PresentationContextID id = findUncompressedPC(UID_FINDStudyRootQueryRetrieveInformationModel, *_scu);
    if (id == 0) {
        MI_IO_LOG(MI_ERROR) << "There is no uncompressed presentation context for Study Root FIND";
        return -1;
    }
    
    OFList<QRResponse*> res;
    OFCondition result = _scu->sendFINDRequest(id,  &query_key, &res);
    if (result.bad()) {
        MI_IO_LOG(MI_ERROR) << "FIND request failed.";
    } else {
        dcm_infos.clear();
        MI_IO_LOG(MI_DEBUG) << "query result size: " << res.size()-1;
        for (auto it = res.begin(); it != res.end(); ++it) {
            if((*it)->m_dataset != NULL) {
                DcmInfo dcm_info;
                fill_dcm_info((*it)->m_dataset, dcm_info, query_level);
                dcm_infos.push_back(dcm_info);
            }
        }
    }
    
    if (!dcm_infos.empty()) {
        _series_to_release_scp = dcm_infos[0].series_id;
    }
    
    return 0;
}

int PACSCommunicator::retrieve_series(const std::string& series_id, const std::string& map_path, std::vector<DcmInstanceInfo>* instance_infos) {
    if(0 != try_connect() ) {
        MI_IO_LOG(MI_FATAL) << "try connect failed before fetch seriess: " << series_id;
        return -1;
    }

    if (!_scp) {
        MI_IO_LOG(MI_ERROR) << "SCP is null when fetch series: " << series_id;
        return -1;
    }
    //set mapping directory
    _scp->setOutputDirectory(map_path.c_str());

    DcmDataset query_key;
    query_key.putAndInsertString(DCM_QueryRetrieveLevel, "SERIES");
    query_key.putAndInsertString(DCM_SeriesInstanceUID,series_id.c_str());

    const T_ASC_PresentationContextID id = findUncompressedPC(UID_MOVEStudyRootQueryRetrieveInformationModel, *_scu);
    if (id == 0) {
        MI_IO_LOG(MI_ERROR) << "There is no uncompressed presentation context for Study Root MOVE";
        return -1;
    }
    
    _scp->set_instance_infos(instance_infos);

    // Test Code
    // OFList<RetrieveResponse*> responses;
    // OFCondition result = _scu->sendMOVERequest(id, _connection_cache->client_ae_title.c_str(), &query_key, &responses);
    // MI_IO_LOG(MI_DEBUG) << "retrieve response size: " << responses.size();
    // for (auto it = responses.begin(); it != responses.end(); ++it) {
    //     MI_IO_LOG(MI_DEBUG) << "RemainingSubops: " << (*it)->m_numberOfFailedSubops 
    //     << ", CompletedSubops: " << (*it)->m_numberOfCompletedSubops 
    //     << ", FailedSubop: " << (*it)->m_numberOfFailedSubops 
    //     << ", WarningSubops: " << (*it)->m_numberOfWarningSubops;
    // }

    OFCondition result = _scu->sendMOVERequest(id, _connection_cache->client_ae_title.c_str(), &query_key, NULL);
    if (result.good() && (!instance_infos || !instance_infos->empty()) ) {
        MI_IO_LOG(MI_DEBUG) << "retrieve series: " << series_id << " success."; 
        return 0;
    } else {
        MI_IO_LOG(MI_ERROR) << "retrieve series: " << series_id << " failed."; 
        return -1;
    }    
}

void PACSCommunicator::run_scp() {
    if (!_connection_cache) {
        MI_IO_LOG(MI_FATAL) << "connection cache is null when run scp.";
        return;
    }
    _scp.reset(new MIDcmSCP());
    _scp->setAETitle(_connection_cache->client_ae_title.c_str());
    _scp->setPort(_connection_cache->client_port);

    OFList<OFString> ts;
    ts.push_back(UID_LittleEndianExplicitTransferSyntax);
    ts.push_back(UID_BigEndianExplicitTransferSyntax);
    ts.push_back(UID_LittleEndianImplicitTransferSyntax);

    // three storage service are needed
    _scp->addPresentationContext(UID_MRImageStorage, ts);
    _scp->addPresentationContext(UID_CTImageStorage, ts);
    _scp->addPresentationContext(UID_SecondaryCaptureImageStorage, ts);

    const OFString file_ext(".dcm");
    _scp->setFilenameExtension(file_ext);
    
    _scp->listen();

    MI_IO_LOG(MI_DEBUG) << "*****************QUIT SCP*****************";
}


MED_IMG_END_NAMESPACE