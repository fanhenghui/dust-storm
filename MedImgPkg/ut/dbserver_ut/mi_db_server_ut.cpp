#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include "log/mi_logger.h"
#include "util/mi_socket_client.h"
#include "util/mi_ipc_common.h"
#include "appcommon/mi_app_common_define.h"
#include "mi_message.pb.h"

using namespace medical_imaging;

int main(int argc , char* argv[]) {
    SocketClient client(INET);
    client.set_server_address("127.0.0.1","8888");
    IPCDataHeader post_header;
    char* post_data = nullptr;
    IPCDataHeader result_header;
    char* result_data = nullptr;

    post_header.msg_id = COMMAND_ID_FE_OPERATION;
    post_header.msg_info1 = OPERATION_ID_QUERY_DICOM;

    MsgString msgSeries;
    msgSeries.set_context("1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405");
    int post_size = msgSeries.ByteSize();
    post_data = new char[post_size];
    if (msgSeries.SerializeToArray(post_data, post_size)) {
        post_header.data_len = post_size;

        if(0 == client.post(post_header, post_data, result_header, result_data) ) {
            std::cout << "get data.";
            //write to disk
            std::fstream out("/home/wangrui22/data/tmp.dcm", std::ios::out | std::ios::binary);
            if (out.is_open()) {
                out.write(result_data, result_header.data_len);
                out.close();
            }
        }
    }
    
    return 0;
}