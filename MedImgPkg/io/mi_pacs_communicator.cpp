#include "mi_pacs_communicator.h"

#include "mi_dcm_scp.h"
#include "mi_dcm_scu.h"
#include "mi_worklist_info.h"


MED_IMG_BEGIN_NAMESPACE

PACSCommunicator::PACSCommunicator() : _initialized(false) {}

PACSCommunicator::~PACSCommunicator() {
  if (this->_scu) {
    this->_scu->endAssociation();
  }
}

bool PACSCommunicator::initialize(const char *self_AE_title,
                                  const unsigned short self_port,
                                  const char *serive_ip_address,
                                  const unsigned short serive_port,
                                  const char *service_AE_title) {
  if (!this->_initialized) {
    this->_scp.reset(new MIDcmSCP(self_AE_title));
    this->_scp->initialize(self_port);

    this->_scu.reset(new MIDcmSCU(self_AE_title));
    if (!this->_scu->createAssociation(serive_ip_address, serive_port,
                                       service_AE_title)) {
      std::cout << "Fail to create association with PACS\n";
      this->_scu.reset(nullptr);
      this->_scp.reset(nullptr);
      return this->_initialized;
    }

    this->_initialized = true;
  }
  return this->_initialized;
}

bool PACSCommunicator::initialize(const char *configure_file_path) {
  OFLog::configure(OFLogger::DEBUG_LOG_LEVEL);
  std::fstream input_file(configure_file_path, std::ios::in);
  if (!input_file.is_open()) {
    std::cout
        << "Cannot open the specified file :(\n try another initializer \n";
    return false;
  } else {
    std::string line;
    std::string tag;
    std::string equal;
    std::string content;

    unsigned int client_port;
    std::string client_title;

    std::string server_host;
    unsigned int server_port;
    std::string server_title;

    while (std::getline(input_file, line)) {
      std::stringstream ss(line);
      ss >> tag >> equal >> content;
      if (tag == std::string("Client_Port")) {
        client_port = stoul(content);
      }

      else if (tag == "Client_Title") {
        client_title = content;
      }

      else if (tag == "Server_Host") {
        server_host = content;
      }

      else if (tag == "Server_Port") {
        server_port = stoul(content);
      }

      else if (tag == "Server_Title") {
        server_title = content;
      }

      else if (tag == "Cache_Path") {
        _cache_path = content;
      }
    }
    input_file.close();

    return this->initialize(client_title.c_str(), client_port,
                            server_host.c_str(), server_port,
                            server_title.c_str());
  }
}

bool PACSCommunicator::populate_whole_work_list() {
  if (!this->_initialized) {
    return false;
  }

  this->_work_list.clear();
  this->_scu->set_work_list(&this->_work_list);
  return this->_scu->search_all();
}

const std::string PACSCommunicator::fetch_dicom(const std::string &series_idx) {
  // try to find the work_list_item
  WorkListInfo *ptr_work_list = nullptr;
  for (int i = 0; i < this->_work_list.size(); ++i) {
    if (series_idx.compare(this->_work_list.at(i).GetSeriesInsUID()) == 0) {
      ptr_work_list = &(this->_work_list.at(i));
      break;
    }
  }
  if (ptr_work_list) {
    return this->fetch_dicom(*ptr_work_list);
  } else
    return "";
}

const std::string PACSCommunicator::fetch_dicom(const WorkListInfo &item) {
  if (!this->_initialized) {
    return "";
  }
  // create/clean a directory which should match the series uid
  OFString directory =
      OFString((_cache_path + "/" + item.GetSeriesInsUID()).c_str());

  // if exist
  if (!OFStandard::dirExists(directory)) {
    // make directory
    OFString root_name(_cache_path.c_str());
    if (OFStandard::createDirectory(directory, root_name).bad()) {
      std::cout << "cannot create directory for downloading images :(\n";
    }
  } else {
    // should we clear this one?! following codes delete the whole folder
    std::cout << "current files in the specified directory will be deleted!\n";
#ifdef WIN32
    std::string cmd = "del /Q \"" + std::string(directory.c_str()) + "\\*.* \"";
#else
    std::string cmd = std::string("rm -r ") + std::string("\"") +
                      std::string(directory.c_str()) + std::string("\"");
#endif
    int ret = system(cmd.c_str());
  }

  OFCondition res = this->_scp->setOutputDirectory(directory.c_str());
  if (res.good() && this->_scu->fetch(this->_scp->getAETitle().c_str(), item)) {
    return std::string(directory.c_str());
  } else {
    return "";
  }
}

MED_IMG_END_NAMESPACE