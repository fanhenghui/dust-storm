#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include "log/mi_logger.h"
#include "util/mi_socket_client.h"
#include "util/mi_ipc_client_proxy.h"
#include "util/mi_ipc_common.h"
#include "io/mi_dicom_loader.h"
#include "appcommon/mi_app_config.h"
#include "appcommon/mi_app_common_define.h"
#include "appcommon/mi_message.pb.h"

using namespace medical_imaging;

class CmdHandlerSendDICOMSeries : public ICommandHandler {
    virtual int handle_command(const IPCDataHeader& dataheader, char* buffer) {
        //just write to disk
        static int id = 0;
        const std::string file_root = "/home/wangrui22/data/test/";
        std::stringstream ss;
        ss << file_root << "DICOM_" << id++ << ".dcm";
        std::fstream out(ss.str(), std::ios::out | std::ios::binary);
        if (out.is_open()) {
            out.write(buffer, dataheader.data_len);
            out.close();
        }

        unsigned int end_tag = dataheader.msg_info2;
        unsigned int dicom_slice = dataheader.msg_info3;

        if (end_tag == 0) {

        }
        return 0;
    };
};

class CmdHandlerBEEnd : public ICommandHandler {
    virtual int handle_command(const IPCDataHeader& dataheader, char* buffer) {
        MI_LOG(MI_INFO) << "[DB UT] " << "in BE end cmd handler.\n";
        return CLIENT_QUIT_ID; 
    };
};

IPCPackage* create_query_dicom_message() {
    IPCDataHeader post_header;
    char* post_data = nullptr;

    post_header.msg_id = COMMAND_ID_BE_DB_OPERATION;
    post_header.msg_info1 = OPERATION_ID_DB_QUERY_DICOM;

    MsgString msgSeries;
    msgSeries.set_context("1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405");
    int post_size = msgSeries.ByteSize();
    post_data = new char[post_size];
    if (!msgSeries.SerializeToArray(post_data, post_size)) {
        MI_LOG(MI_ERROR) << "[DB UT] " << "serialize message failed.\n";
        return nullptr;
    }
    post_header.data_len = post_size;
    return (new IPCPackage(post_header,post_data));
}

IPCPackage* create_query_preprocess_mask_message() {
    IPCDataHeader post_header;
    char* post_data = nullptr;

    post_header.msg_id = COMMAND_ID_BE_DB_OPERATION;
    post_header.msg_info1 = OPERATION_ID_DB_QUERY_PREPROCESS_MASK;

    MsgString msgSeries;
    msgSeries.set_context("1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405");
    int post_size = msgSeries.ByteSize();
    post_data = new char[post_size];
    if (!msgSeries.SerializeToArray(post_data, post_size)) {
        MI_LOG(MI_ERROR) << "[DB UT] " << "serialize message failed.\n";
        return nullptr;
    }
    post_header.data_len = post_size;
    return (new IPCPackage(post_header,post_data));
}

IPCPackage* create_query_ai_annotation_message() {
    IPCDataHeader post_header;
    char* post_data = nullptr;

    post_header.msg_id = COMMAND_ID_BE_DB_OPERATION;
    post_header.msg_info1 = OPERATION_ID_DB_QUERY_AI_ANNOTATION;

    MsgString msgSeries;
    msgSeries.set_context("1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405");
    int post_size = msgSeries.ByteSize();
    post_data = new char[post_size];
    if (!msgSeries.SerializeToArray(post_data, post_size)) {
        MI_LOG(MI_ERROR) << "[DB UT] " << "serialize message failed.\n";
        return nullptr;
    }
    post_header.data_len = post_size;
    return (new IPCPackage(post_header,post_data));
}

IPCPackage* create_query_end_message() {
    IPCDataHeader header;
    header.msg_id = COMMAND_ID_BE_DB_OPERATION;
    header.msg_info1 = OPERATION_ID_DB_QUERY_END;
    return (new IPCPackage(header));
}

int main(int argc , char* argv[]) {
    const std::string log_config_file = AppConfig::instance()->get_log_config_file();
    Logger::instance()->bind_config_file(log_config_file);
    Logger::instance()->set_file_name_format("logs/mi-db-ut-%Y-%m-%d_%H-%M-%S.%N.log");
    Logger::instance()->set_file_direction("");
    Logger::instance()->initialize();

    IPCClientProxy client_proxy(INET);
    client_proxy.set_server_address("127.0.0.1","8888");
    client_proxy.register_command_handler(COMMAND_ID_DB_SEND_DICOM_SERIES, 
    std::shared_ptr<CmdHandlerSendDICOMSeries>(new CmdHandlerSendDICOMSeries()));
    client_proxy.register_command_handler(COMMAND_ID_DB_SEND_END, 
    std::shared_ptr<CmdHandlerBEEnd>(new CmdHandlerBEEnd()));
    
    std::vector<IPCPackage*> packages;
    packages.push_back(create_query_dicom_message());
    packages.push_back(create_query_end_message());

    if(0 == client_proxy.sync_post(packages) ) {
        std::cout << "sync post success.\n";
    } else {
        std::cout << "sync post failed.\n";
    }
    
    return 0;
}