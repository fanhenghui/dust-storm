#include "mi_pacs_communicator.h"
#include "mi_dcm_scu.h"
#include "mi_dcm_scp.h"
#include "mi_io_logger.h"

MED_IMG_BEGIN_NAMESPACE

//官方demo只给了没有压缩的三种传输类型，实际中应该如何处理？
static Uint8 findUncompressedPC(const OFString& sopClass, DcmSCU& scu) { 
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

static void fill_dcm_info(DcmDataset* dataset, DcmInfo& info) {
    OFString str;
    if (dataset->findAndGetOFString(DCM_StudyDate, str).good()) {
        info.study_date = std::string(str.c_str());
    }
    if (dataset->findAndGetOFString(DCM_StudyTime, str).good()) {
        info.study_time = std::string(str.c_str());
    }
    if (dataset->findAndGetOFString(DCM_StudyInstanceUID, str).good()) {
        info.study_id = std::string(str.c_str());
    }
    if (dataset->findAndGetOFString(DCM_SeriesInstanceUID, str).good()) {
        info.series_id = std::string(str.c_str());
    }
    if (dataset->findAndGetOFString(DCM_StudyDescription, str).good()) {
        info.study_description = std::string(str.c_str());
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
    if (dataset->findAndGetOFString(DCM_PatientAge, str).good()) {
        info.patient_age = std::string(str.c_str());
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

    _scp_thread = boost::thread(boost::bind(&PACSCommunicator::run_scp_i, this));
    return try_connect_i();
}

void PACSCommunicator::disconnect() {
    if (_scp && _scu) {
        _scp->stop();

        DcmDataset query_key;
        query_key.putAndInsertString(DCM_QueryRetrieveLevel, "SERIES");
        query_key.putAndInsertString(DCM_SeriesInstanceUID, _series_to_release_scp.c_str());

        const T_ASC_PresentationContextID id = findUncompressedPC(UID_MOVEStudyRootQueryRetrieveInformationModel, *_scu);
        if (id == 0) {
            MI_IO_LOG(MI_ERROR) << "There is no uncompressed presentation context for Study Root MOVE";
        } else {
            //send last message to call SCP's stopAfterCurrentAssociation to return true , then stop listen
            //TODO 这里是有风险的：取了最近一次query all series 的第一个series，万一move成功，就不会走到stopAfterCurrentAssociation函数，因此没法关闭SCP
            MI_IO_LOG(MI_INFO) << "plase wait last query message.";
            _scu->sendMOVERequest(id, _connection_cache->client_ae_title.c_str(), &query_key, NULL);
            _scp_thread.join();        
        }
    }
    if (_scu) {
        _scu->closeAssociation(DCMSCU_RELEASE_ASSOCIATION);
    }
}

int PACSCommunicator::try_connect_i() {
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

    result = _scu->negotiateAssociation();
    if (result.bad()) {
        MI_IO_LOG(MI_FATAL) << "SCU negotiate association failed.";
        return -1;
    }

    result = _scu->sendECHORequest(0);
    if (result.bad()) {
        MI_IO_LOG(MI_FATAL) << "SCU echo server failed.";
        return -1;
    }

    return 0;
}

int PACSCommunicator::retrieve_all_series(std::vector<DcmInfo>& dcm_infos) {
    if(0 != try_connect_i() ) {
        MI_IO_LOG(MI_FATAL) << "try connect failed before query all series.";
        return -1;
    }

    DcmDataset query_key;
    query_key.putAndInsertString(DCM_QueryRetrieveLevel, "SERIES");
    query_key.putAndInsertString(DCM_StudyDate, "");
    query_key.putAndInsertString(DCM_StudyTime, "");
    query_key.putAndInsertString(DCM_StudyInstanceUID, "");
    query_key.putAndInsertString(DCM_SeriesInstanceUID, "");
    query_key.putAndInsertString(DCM_Modality, "");
    query_key.putAndInsertString(DCM_PatientID, "");
    query_key.putAndInsertString(DCM_PatientName, "");
    query_key.putAndInsertString(DCM_PatientSex, "");
    query_key.putAndInsertString(DCM_PatientBirthDate, "");
    query_key.putAndInsertString(DCM_PatientAge, "");
    query_key.putAndInsertString(DCM_SeriesNumber, "");

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
        MI_IO_LOG(MI_DEBUG) << "series size: " << res.size();
        for (auto it = res.begin(); it != res.end(); ++it) {
            if((*it)->m_dataset != NULL) {
                DcmInfo dcm_info;
                fill_dcm_info((*it)->m_dataset, dcm_info);
                dcm_infos.push_back(dcm_info);
            }
        }
    }
    
    if (!dcm_infos.empty()) {
        _series_to_release_scp = dcm_infos[0].series_id;
    }
    
    return 0;
}

int PACSCommunicator::fetch_series(const std::string& series_id, const std::string& map_path) {
    if(0 != try_connect_i() ) {
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
    
    OFCondition result = _scu->sendMOVERequest(id, _connection_cache->client_ae_title.c_str(), &query_key, NULL);
    if (result.bad()) {
        MI_IO_LOG(MI_ERROR) << "retrieve series: " << series_id << " failed."; 
        return -1;
    } else {
        MI_IO_LOG(MI_DEBUG) << "retrieve series: " << series_id << " success."; 
        return 0;
    }
}

void PACSCommunicator::run_scp_i() {
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