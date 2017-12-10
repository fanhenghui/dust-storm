#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include "log/mi_logger.h"
#include "util/mi_socket_client.h"
#include "util/mi_ipc_client_proxy.h"
#include "util/mi_ipc_common.h"
#include "util/mi_memory_shield.h"
#include "io/mi_dicom_loader.h"
#include "arithmetic/mi_run_length_operator.h"
#include "io/mi_configure.h"
#include "appcommon/mi_app_common_define.h"
#include "io/mi_message.pb.h"

using namespace medical_imaging;
class CmdHandlerDBSendDICOMSeries : public ICommandHandler {
    virtual int handle_command(const IPCDataHeader& dataheader, char* buffer) {
        MI_LOG(MI_INFO) << "[DB UT] " << "IN DICOM series handler.";
        MemShield shield(buffer);

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

        // unsigned int end_tag = dataheader.reserved0;
        // unsigned int dicom_slice = dataheader.reserved1;

        // if (end_tag == 0) {

        // }

        return 0;
    };
};

class CmdHandlerDBSendAIAnnotation : public ICommandHandler {
    virtual int handle_command(const IPCDataHeader& dataheader, char* buffer) {
        MI_LOG(MI_INFO) << "[DB UT] " << "IN AI annotation handler.";
        MemShield shield(buffer);
        //just write to disk
        if (nullptr == buffer) {
            MI_LOG(MI_INFO) << "[DB UT] " << "DB send AI annotation buffer null.";
            return -1;
        }
        MsgAnnotationCollectionDB msgAnnos;
        if (!msgAnnos.ParseFromArray(buffer,dataheader.data_len)) {
            MI_LOG(MI_INFO) << "[DB UT] " << "parse AI annotation buffer failed.";
            return -1;
        }

        MI_LOG(MI_INFO) << "[DB UT] " << "AI annotation series: " << msgAnnos.series_uid();
        for (int i = 0; i < msgAnnos.annotation_size(); ++i) {
            const MsgAnnotationUnitDB& anno = msgAnnos.annotation(i);
            MI_LOG(MI_INFO) << "[DB UT] " << "(" << anno.x() << "," << anno.y() << "," << anno.z() << ") " 
            << anno.r() << ", " << anno.p();
        }
        msgAnnos.Clear();
        return 0;
    };
};

class CmdHandlerDBSendPreprocessMask : public ICommandHandler {
    virtual int handle_command(const IPCDataHeader& dataheader, char* buffer) {
        MI_LOG(MI_INFO) << "[DB UT] " << "IN preprocess mask handler.";
        MemShield shield(buffer);
        if (nullptr == buffer) {
            MI_LOG(MI_INFO) << "[DB UT] " << "DB send preprocess mask buffer null.";
            return -1;
        }
        //just write to disk
        const std::string file_path = "/home/wangrui22/data/test/1.rle";
        std::fstream out(file_path, std::ios::out | std::ios::binary);
        if (out.is_open()) {
            out.write(buffer, dataheader.data_len);
            out.close();
        }

        unsigned char* buffer2 = new unsigned char[512*512*312];
        if (0 != RunLengthOperator::decode(file_path,buffer2,512*512*312)) {
            MI_LOG(MI_INFO) << "[DB UT] " << "decode rle failed.";
            return -1;
        }
        return 0;
    };
};

class CmdHandlerDBSendEnd : public ICommandHandler {
    virtual int handle_command(const IPCDataHeader& dataheader, char* buffer) {
        MI_LOG(MI_INFO) << "[DB UT] " << "in BE end cmd handler.";
        return CLIENT_QUIT_ID; 
    };
};

class CmdHandlerDBSendError : public ICommandHandler {
    virtual int handle_command(const IPCDataHeader& dataheader, char* buffer) {
        MI_LOG(MI_INFO) << "[DB UT] " << "IN Error handler.";
        if (nullptr == buffer) {
            MI_LOG(MI_ERROR) << "[DB UT] " << "DB server error has no message.";
            return -1;
        }
        MsgString msgErr;
        if(!msgErr.ParseFromArray(buffer, dataheader.data_len)) {
            MI_LOG(MI_ERROR) << "[DB UT] " << "DB server error has no message.";
            return -1;
        } else {
            const std::string err = msgErr.context();
            msgErr.Clear();
            MI_LOG(MI_ERROR) << "[DB UT] " << "DB server error message: " << err;
        }

        return 0; 
    };
};

const static std::string series_id = "1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405";
IPCPackage* create_query_dicom_message() {
    IPCDataHeader post_header;
    char* post_data = nullptr;

    post_header.msg_id = COMMAND_ID_DB_BE_OPERATION;
    post_header.op_id = OPERATION_ID_DB_BE_FETCH_DICOM;

    MsgString msgSeries;
    msgSeries.set_context(series_id);
    int post_size = msgSeries.ByteSize();
    post_data = new char[post_size];
    if (!msgSeries.SerializeToArray(post_data, post_size)) {
        MI_LOG(MI_ERROR) << "[DB UT] " << "serialize message failed.";
        return nullptr;
    }
    post_header.data_len = post_size;
    return (new IPCPackage(post_header,post_data));
}

IPCPackage* create_query_preprocess_mask_message() {
    IPCDataHeader post_header;
    char* post_data = nullptr;

    post_header.msg_id = COMMAND_ID_DB_BE_OPERATION;
    post_header.op_id = OPERATION_ID_DB_BE_FETCH_PREPROCESS_MASK;

    MsgString msgSeries;
    msgSeries.set_context(series_id);
    int post_size = msgSeries.ByteSize();
    post_data = new char[post_size];
    if (!msgSeries.SerializeToArray(post_data, post_size)) {
        MI_LOG(MI_ERROR) << "[DB UT] " << "serialize message failed.";
        return nullptr;
    }
    post_header.data_len = post_size;
    return (new IPCPackage(post_header,post_data));
}

IPCPackage* create_query_ai_annotation_message() {
    IPCDataHeader post_header;
    char* post_data = nullptr;

    post_header.msg_id = COMMAND_ID_DB_BE_OPERATION;
    post_header.op_id = OPERATION_ID_DB_BE_FETCH_AI_EVALUATION;

    MsgString msgSeries;
    msgSeries.set_context(series_id);
    int post_size = msgSeries.ByteSize();
    post_data = new char[post_size];
    if (!msgSeries.SerializeToArray(post_data, post_size)) {
        MI_LOG(MI_ERROR) << "[DB UT] " << "serialize message failed.";
        return nullptr;
    }
    post_header.data_len = post_size;
    return (new IPCPackage(post_header,post_data));
}

IPCPackage* create_query_end_message() {
    IPCDataHeader header;
    header.msg_id = COMMAND_ID_DB_BE_OPERATION;
    header.op_id = OPERATION_ID_DB_BE_REQUEST_END;
    return (new IPCPackage(header));
}

int main(int argc , char* argv[]) {
    const std::string log_config_file = Configure::instance()->get_log_config_file();
    Logger::instance()->bind_config_file(log_config_file);
    Logger::instance()->set_file_name_format("logs/mi-db-ut-%Y-%m-%d_%H-%M-%S.%N.log");
    Logger::instance()->set_file_direction("");
    Logger::instance()->initialize();

    IPCClientProxy client_proxy(INET);
    client_proxy.set_server_address("127.0.0.1","8888");
    client_proxy.register_command_handler(COMMAND_ID_BE_DB_SEND_DICOM_SERIES, 
    std::shared_ptr<CmdHandlerDBSendDICOMSeries>(new CmdHandlerDBSendDICOMSeries()));
    client_proxy.register_command_handler(COMMAND_ID_BE_DB_SEND_PREPROCESS_MASK, 
    std::shared_ptr<CmdHandlerDBSendPreprocessMask>(new CmdHandlerDBSendPreprocessMask()));
    client_proxy.register_command_handler(COMMAND_ID_BE_DB_SEND_AI_EVALUATION, 
    std::shared_ptr<CmdHandlerDBSendAIAnnotation>(new CmdHandlerDBSendAIAnnotation()));
    
    client_proxy.register_command_handler(COMMAND_ID_BE_DB_SEND_END, 
    std::shared_ptr<CmdHandlerDBSendEnd>(new CmdHandlerDBSendEnd()));
    client_proxy.register_command_handler(COMMAND_ID_BE_DB_SEND_ERROR, 
    std::shared_ptr<CmdHandlerDBSendError>(new CmdHandlerDBSendError()));
    
    std::vector<IPCPackage*> packages;
    packages.push_back(create_query_dicom_message());
    packages.push_back(create_query_preprocess_mask_message());
    packages.push_back(create_query_ai_annotation_message());
    packages.push_back(create_query_end_message());

    if(0 == client_proxy.sync_post(packages) ) {
        MI_LOG(MI_INFO) << "[DB UT] "  << "sync post success.";
    } else {
        MI_LOG(MI_ERROR) << "[DB UT] "  << "sync post failed.";
    }
    
    return 0;
}