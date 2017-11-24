#ifndef MED_IMG_DB_SERVER_COMMON_H
#define MED_IMG_DB_SERVER_COMMON_H

#include "util/mi_exception.h"
#include "util/mi_ipc_common.h"
#include "appcommon/mi_app_common_define.h"
#include "med_img_pkg_config.h"

MED_IMG_BEGIN_NAMESPACE

#ifndef DBSERVER_THROW_EXCEPTION
#define DBSERVER_THROW_EXCEPTION(desc) THROW_EXCEPTION("DBS", desc);
#endif

#ifndef DBSERVER_CHECK_NULL_EXCEPTION
#define DBSERVER_CHECK_NULL_EXCEPTION(pointer)                                   \
  if (nullptr == pointer) {                                                    \
    DBSERVER_THROW_EXCEPTION(std::string(typeid(pointer).name()) +               \
                           std::string(" ") + std::string(#pointer) +          \
                           " is null.");                                       \
  }
#endif

#ifndef SEND_ERROR_TO_BE
#define SEND_ERROR_TO_BE(server_proxy, client_socket_id, err)                   \
if (nullptr != server_proxy && !std::string(err).empty()) {             \
    MsgString msgErr;                                                   \
    msgErr.set_context(err);                                            \
    const int buffer_size = msgErr.ByteSize();                          \
    if (buffer_size > 0) {                                              \
        IPCDataHeader header;                                           \
        header.receiver = (client_socket_id);                                   \
        header.msg_id = COMMAND_ID_DB_SEND_ERROR;                       \
        header.data_len = buffer_size;                                  \
        char* buffer = new char[buffer_size];                           \
        if (nullptr != buffer) {                                        \
            if (!msgErr.SerializeToArray(buffer, buffer_size)) {        \
                delete [] buffer;                                       \
                buffer = nullptr;                                       \
            } else {                                                    \
                IPCPackage* package = new IPCPackage(header, buffer);   \
                if (0 != server_proxy->async_send_data(package)) {      \
                    delete package;                                     \
                    package = nullptr;                                  \
                }                                                       \
            }                                                           \
        }                                                               \
    }                                                                   \
}                                                                       
#endif

MED_IMG_END_NAMESPACE

#endif