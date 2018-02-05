#ifndef MEDIMGIO_PACS_COMMUNICATOR_H
#define MEDIMGIO_PACS_COMMUNICATOR_H

#include <string>
#include <vector>
#include <memory>
#include <boost/thread/thread.hpp>

#include "io/mi_io_export.h"
#include "io/mi_dicom_info.h"
#include "io/mi_io_define.h"

MED_IMG_BEGIN_NAMESPACE

class MIDcmSCU;
class MIDcmSCP;
class IO_Export PACSCommunicator {
public:
    explicit PACSCommunicator(bool open_dcmtk_console_log=false);
    ~PACSCommunicator();

    int connect(const std::string& server_ae_title,const std::string& server_host, unsigned short server_port,
                const std::string& client_ae_title, unsigned short client_port);
    void disconnect();

    int query_patient(const PatientInfo& patient_key, std::vector<PatientInfo>* patient_infos);
    int query_study(const PatientInfo& patient_key, const StudyInfo& study_key, std::vector<PatientInfo>* patient_infos, std::vector<StudyInfo>* study_infos);
    int query_series(const PatientInfo& patient_key, const StudyInfo& study_key, const SeriesInfo& series_key,   
        std::vector<PatientInfo>* patient_infos, std::vector<StudyInfo>* study_infos, std::vector<SeriesInfo>* series_infos);

    int retrieve_series(const std::string& series_id, const std::string& map_path, std::vector<InstanceInfo>* instance_infos = nullptr);

private:
    int try_connect();
    void run_scp();
    void run_scu_cancel(int timeout, int pres_id);

private:
    struct ConnectionCache;
    std::unique_ptr<ConnectionCache> _connection_cache;
    std::unique_ptr<MIDcmSCP> _scp;
    std::unique_ptr<MIDcmSCU> _scu;

    std::string _series_to_release_scp;
    bool _querying_release_series;

    boost::thread _scp_thread;

private:
    DISALLOW_COPY_AND_ASSIGN(PACSCommunicator);
};

MED_IMG_END_NAMESPACE
#endif