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

struct QueryKey {
    std::string study_date;
    std::string patient_id;
    std::string patient_name;
    std::string modality;
    std::string accession_number;
    std::string patient_sex;
};

class MIDcmSCU;
class MIDcmSCP;
class IO_Export PACSCommunicator {
public:
    explicit PACSCommunicator(bool open_dcmtk_console_log=false);
    ~PACSCommunicator();

    int connect(const std::string& server_ae_title,const std::string& server_host, unsigned short server_port,
                const std::string& client_ae_title, unsigned short client_port);
    void disconnect();

    int query_series(std::vector<DcmInfo>& dcm_infos, const QueryKey& key);
    int retrieve_series(const std::string& series_id, const std::string& map_path);

private:
    int try_connect();
    void run_scp();

private:
    struct ConnectionCache;
    std::unique_ptr<ConnectionCache> _connection_cache;
    std::unique_ptr<MIDcmSCP> _scp;
    std::unique_ptr<MIDcmSCU> _scu;
    std::string _series_to_release_scp;

    boost::thread _scp_thread;

private:
    DISALLOW_COPY_AND_ASSIGN(PACSCommunicator);
};

MED_IMG_END_NAMESPACE
#endif