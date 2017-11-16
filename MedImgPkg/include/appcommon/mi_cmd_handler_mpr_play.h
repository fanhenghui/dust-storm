// #ifndef MED_IMG_APPCOMMON_MI_CMD_HANDLER_MPR_PLAY_H
// #define MED_IMG_APPCOMMON_MI_CMD_HANDLER_MPR_PLAY_H

// #include <memory>
// #include "util/mi_ipc_common.h"
// #include "appcommon/mi_app_common_export.h"
// #include "appcommon/mi_operation_interface.h"


// MED_IMG_BEGIN_NAMESPACE

// class AppController;
// class AppCommon_Export CmdHandlerMPRPlay : public ICommandHandler {
// public:
//     CmdHandlerMPRPlay(std::shared_ptr<AppController> controller);

//     virtual ~CmdHandlerMPRPlay();

//     virtual int handle_command(const IPCDataHeader& dataheader , char* buffer);

// private:
//     void logic_i(OpDataHeader& op_header, char* buffer);

// private:
//     std::weak_ptr<AppController> _controller;
//     bool _playing;

// };

// MED_IMG_END_NAMESPACE

// #endif