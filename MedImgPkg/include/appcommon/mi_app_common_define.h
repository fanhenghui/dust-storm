#ifndef MED_IMG_APPCOMMON_MI_APP_COMMON_DEFINE_H
#define MED_IMG_APPCOMMON_MI_APP_COMMON_DEFINE_H

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
#define COMMAND_ID_FE_MPR_PLAY 120003
#define COMMAND_ID_FE_VR_PLAY 120004

// BE to FE
#define COMMAND_ID_BE_HEARTBEAT 269999
#define COMMAND_ID_BE_READY 270000
#define COMMAND_ID_BE_SEND_IMAGE 270001
#define COMMAND_ID_BE_SEND_WORKLIST 270002
#define COMMAND_ID_BE_SEND_ANNOTATION 270003
#define COMMAND_ID_BE_SEND_NONE_IMAGE 270004

// FE to BE OPERATION ID
#define OPERATION_ID_MPR_PAGING 310001
#define OPERATION_ID_PAN 310002
#define OPERATION_ID_ZOOM 310003
#define OPERATION_ID_ROTATE 310004
#define OPERATION_ID_WINDOWING 310005
#define OPERATION_ID_RESIZE 310006

MED_IMG_END_NAMESPACE

#endif