#ifndef MEDIMGIO_PACS_COMMUNICATOR_H
#define MEDIMGIO_PACS_COMMUNICATOR_H

#include "io/mi_io_export.h"
#include <memory>
#include <string>
#include <vector>


MED_IMG_BEGIN_NAMESPACE

class MIDcmSCP;
class MIDcmSCU;
class WorkListInfo;

class IO_Export PACSCommunicator {
public:
    PACSCommunicator();
    ~PACSCommunicator();

    bool initialize(const char* configure_file_path);
    bool initialize(const char* self_AE_title, const unsigned short self_port,
                    const char* serive_ip_address,
                    const unsigned short serive_port,
                    const char* service_AE_title);

    bool populate_whole_work_list();
    const std::vector<WorkListInfo>& get_work_list() {
        return _work_list;
    };

    const std::string fetch_dicom(const std::string& series_idx);
    const std::string fetch_dicom(const WorkListInfo& item);

private:
    bool _initialized;
    std::unique_ptr<MIDcmSCP> _scp;
    std::unique_ptr<MIDcmSCU> _scu;
    std::vector<WorkListInfo> _work_list;
    std::string _cache_path;
};

MED_IMG_END_NAMESPACE
#endif
