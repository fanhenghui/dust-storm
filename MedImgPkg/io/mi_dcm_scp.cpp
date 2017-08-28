#include "mi_dcm_scp.h"

#include "boost/thread.hpp"

MED_IMG_BEGIN_NAMESPACE

MIDcmSCP::MIDcmSCP(const char * self_AE_title)
{
    this->setAETitle(self_AE_title);

    OFList<OFString> ts;
    ts.push_back(UID_LittleEndianExplicitTransferSyntax);
    ts.push_back(UID_BigEndianExplicitTransferSyntax);
    ts.push_back(UID_LittleEndianImplicitTransferSyntax);

    // three storage service are needed
    this->addPresentationContext(UID_MRImageStorage, ts);
    this->addPresentationContext(UID_CTImageStorage, ts);
    this->addPresentationContext(UID_SecondaryCaptureImageStorage, ts);

    // output settings
    //DcmStorageSCP::E_DirectoryGenerationMode opt_directoryGeneration = DcmStorageSCP::DGM_NoSubdirectory;
    //this->setDirectoryGenerationMode(opt_directoryGeneration);
    //DcmStorageSCP::E_FilenameGenerationMode opt_filenameGeneration = DcmStorageSCP::FGM_SOPInstanceUID;
    //this->setFilenameGenerationMode(opt_filenameGeneration);
    //DcmStorageSCP::E_DatasetStorageMode opt_datasetStorage = DcmStorageSCP::DGM_StoreToFile;
    //this->setDatasetStorageMode(opt_datasetStorage);
    const char *opt_filenameExtension = ".dcm";
    this->setFilenameExtension(opt_filenameExtension);
}

void MIDcmSCP::initialize(const unsigned short port)
{
    this->setPort(port);
    boost::thread listener(&MIDcmSCP::start_listening, this, port);
    listener.detach();
}

void MIDcmSCP::start_listening(const unsigned short port)
{
    this->setPort(port);
    this->listen();
}

OFCondition MIDcmSCP::handleIncomingCommand(T_DIMSE_Message *incomingMsg, const DcmPresentationContextInfo &presInfo)
{
    return this->DcmStorageSCP::handleIncomingCommand(incomingMsg, presInfo);
}

//OFCondition MIDcmSCP::handleSTORERequest(T_DIMSE_C_StoreRQ &reqMessage, const T_ASC_PresentationContextID presID, DcmDataset *&reqDataset)
//{
//    //write to the folder specified by _output_directory
//    std::cout << "handle \n";
//    return this->MIDcmSCP::handleSTORERequest(reqMessage, presID, reqDataset);
//}

MED_IMG_END_NAMESPACE