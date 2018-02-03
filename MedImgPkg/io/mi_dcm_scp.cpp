#include "mi_dcm_scp.h"
#include "mi_io_logger.h"

#include "util/mi_file_util.h"

MED_IMG_BEGIN_NAMESPACE

MIDcmSCP::MIDcmSCP():_stop(false),_instance_infos(nullptr) {
}

MIDcmSCP::~MIDcmSCP() {
}

void MIDcmSCP::stop() {
    _stop = true;
}

OFCondition MIDcmSCP::handleIncomingCommand(T_DIMSE_Message* incomingMsg,
                                const DcmPresentationContextInfo& presInfo) {
    return DcmStorageSCP::handleIncomingCommand(incomingMsg, presInfo);
}

OFBool MIDcmSCP::stopAfterCurrentAssociation() {
    if (_stop) {
        MI_IO_LOG(MI_INFO) << "make SCP listen stop after current association.";
    }
    return _stop;
}

Uint16 MIDcmSCP::checkAndProcessSTORERequest(const T_DIMSE_C_StoreRQ &reqMessage, DcmFileFormat &fileformat) {
    DcmDataset *dataset = fileformat.getDataset();
    if ((dataset != NULL) && !dataset->isEmpty()) {
        OFString filename;
        OFString directoryName;
        OFString sopClassUID = reqMessage.AffectedSOPClassUID;
        OFString sopInstanceUID = reqMessage.AffectedSOPInstanceUID;
        if (generateDirAndFilename(filename, directoryName, sopClassUID, sopInstanceUID, dataset).good()) {
            if (_instance_infos) {
                _instance_infos->push_back(DcmInstanceInfo(sopClassUID.c_str(), sopInstanceUID.c_str(), filename.c_str()));
            }
        }

        if (_instance_infos) {
            const DcmInstanceInfo& info = (*_instance_infos)[_instance_infos->size()-1];
            MI_IO_LOG(MI_DEBUG) << "SCP process one store instance: " << info.sop_class_uid << ", " 
                << info.sop_instance_uid << ", " << info.file_path;
        }

        Uint16 statusCode = DcmStorageSCP::checkAndProcessSTORERequest(reqMessage, fileformat);
        if (STATUS_Success == statusCode) {
            int64_t file_size = 0;
            if( -1 != FileUtil::get_file_size(filename.c_str(), file_size)) {
                (*_instance_infos)[_instance_infos->size()-1].file_size = file_size;
            }
        }

        return statusCode;
    } else {
        return STATUS_STORE_Error_CannotUnderstand;
    }
}

void MIDcmSCP::set_instance_infos(std::vector<DcmInstanceInfo>* instance_infos) {
    _instance_infos = instance_infos;
}

MED_IMG_END_NAMESPACE