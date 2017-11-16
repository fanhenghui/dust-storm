#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include "log/mi_logger.h"
#include "util/mi_socket_client.h"
#include "util/mi_ipc_client_proxy.h"
#include "util/mi_ipc_common.h"
#include "io/mi_dicom_loader.h"
#include "appcommon/mi_app_common_define.h"
#include "mi_message.pb.h"

using namespace medical_imaging;

class CmdHandlerReceiveDICOMSeries : public ICommandHandler {
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
            return 0;
        } else {
           return CLIENT_QUIT_ID; 
        }
    };
};

int main(int argc , char* argv[]) {
    IPCClientProxy client_proxy(INET);
    client_proxy.set_server_address("127.0.0.1","8888");
    client_proxy.register_command_handler(COMMAND_ID_BE_RECEIVE_DICOM_SERIES, 
    std::shared_ptr<CmdHandlerReceiveDICOMSeries>(new CmdHandlerReceiveDICOMSeries()));

    std::vector<IPCPackage*> packages;

    IPCDataHeader post_header;
    char* post_data = nullptr;

    post_header.msg_id = COMMAND_ID_FE_OPERATION;
    post_header.msg_info1 = OPERATION_ID_QUERY_DICOM;

    MsgString msgSeries;
    msgSeries.set_context("1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405");
    int post_size = msgSeries.ByteSize();
    post_data = new char[post_size];
    if (!msgSeries.SerializeToArray(post_data, post_size)) {
        std::cout << "serialize message failed.\n";
        return -1;
    }
    post_header.data_len = post_size;

    //shutdown message

    packages.push_back(new IPCPackage(post_header,post_data));

    if(0 == client_proxy.sync_post(packages) ) {
        std::cout << "sync post success.\n";
    } else {
        std::cout << "sync post failed.\n";
    }
    
    return 0;
}