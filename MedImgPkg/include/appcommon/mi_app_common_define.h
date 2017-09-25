#ifndef NED_IMG_APP_COMMON_DEFINE_H_
#define NED_IMG_APP_COMMON_DEFINE_H_

#include "appcommon/mi_app_common_export.h"

MED_IMG_BEGIN_NAMESPACE

// GL Context
#define MAIN_CONTEXT 0
#define RENDERING_CONTEXT 1
#define OPERATION_CONTEXT 2

// FE to BE
#define COMMAND_ID_FE_HEARTBEAT 119999
#define COMMAND_ID_FE_SHUT_DOWN 120000
#define COMMAND_ID_FE_READY 120001
#define COMMAND_ID_FE_OPERATION 120002

// BE to FE
#define COMMAND_ID_BE_HEARTBEAT 269999
#define COMMAND_ID_BE_READY 270000
#define COMMAND_ID_BE_SEND_IMAGE 270001
#define COMMAND_ID_BE_SEND_WORKLIST 270002
#define COMMAND_ID_BE_SEND_ANNOTATION 270003
MED_IMG_END_NAMESPACE

#endif