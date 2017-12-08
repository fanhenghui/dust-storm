#ifndef MED_IMG_APPCOMMON_MI_APP_COMMON_DEFINE_H
#define MED_IMG_APPCOMMON_MI_APP_COMMON_DEFINE_H

#include "appcommon/mi_app_common_export.h"

MED_IMG_BEGIN_NAMESPACE

//***************************************************************//
// GL Context
#define MAIN_CONTEXT 0
#define RENDERING_CONTEXT 1
#define OPERATION_CONTEXT 2
//***************************************************************//

//***************************************************************//
//Command/Operation ID format: {COMMAND, OPERATION}_ID_{receiver}_{sender}_{logic description}
//
// BE receive FE command
#define COMMAND_ID_BE_FE_HEARTBEAT 119999
#define COMMAND_ID_BE_FE_SHUTDOWN 120000
#define COMMAND_ID_BE_FE_READY 120001
#define COMMAND_ID_BE_FE_OPERATION 120002
#define COMMAND_ID_BE_FE_PLAY_MPR 120003
#define COMMAND_ID_BE_FE_PLAY_VR 120004
#define COMMAND_ID_BE_FE_DB_RETRIEVE 120005
#define COMMAND_ID_BE_FE_BACK_TO_WORKLIST 120006
#define COMMAND_ID_BE_FE_PACS_RETRIEVE 120007
#define COMMAND_ID_BE_FE_PACS_FETCH 120008

// FE receive BE command
#define COMMAND_ID_FE_BE_HEARTBEAT 269999
#define COMMAND_ID_FE_BE_READY 270000
#define COMMAND_ID_FE_BE_SEND_IMAGE 270001
#define COMMAND_ID_FE_BE_DB_RETRIEVE_RESULT 270002
#define COMMAND_ID_FE_BE_SEND_ANNOTATION_LIST 270003
#define COMMAND_ID_FE_BE_SEND_NONE_IMAGE 270004
#define COMMAND_ID_FE_BE_PACS_RETRIEVE_RESULT 270005
#define COMMAND_ID_FE_BE_PACS_FETCH_RESULT 270006

// BE receive FE operation
#define OPERATION_ID_BE_FE_INIT 310000
#define OPERATION_ID_BE_FE_MPR_PAGING 310001
#define OPERATION_ID_BE_FE_PAN 310002
#define OPERATION_ID_BE_FE_ZOOM 310003
#define OPERATION_ID_BE_FE_ROTATE 310004
#define OPERATION_ID_BE_FE_WINDOWING 310005
#define OPERATION_ID_BE_FE_RESIZE 310006
#define OPERATION_ID_BE_FE_ANNOTATION 310007
#define OPERATION_ID_BE_FE_MPR_MASK_OVERLAY 310008
#define OPERATION_ID_BE_FE_SWITCH_PRESET_WINDOWING 310009
#define OPERATION_ID_BE_FE_SWITCH_PRESET_VRT 310010
#define OPERATION_ID_BE_FE_LOCATE 310011
#define OPERATION_ID_BE_FE_DOWNSAMPLE 310012
#define OPERATION_ID_BE_FE_FETCH_AI_EVALUATION 310013

//DB receive BE operation
#define COMMAND_ID_DB_BE_OPERATION 410001
//DB receive AI operation
#define COMMAND_ID_DB_AI_OPERATION 410002
//AI receive DB operation
#define COMMAND_ID_AI_DB_OPERATION 410003

//DB receive BE operation
#define OPERATION_ID_DB_BE_FETCH_DICOM 510001
#define OPERATION_ID_DB_BE_FETCH_PREPROCESS_MASK 510002
#define OPERATION_ID_DB_BE_FETCH_AI_EVALUATION 510003
#define OPERATION_ID_DB_BE_REQUEST_END 510004
#define OPERATION_ID_DB_BE_PACS_RETRIEVE 510005
#define OPERATION_ID_DB_BE_PACS_FETCH 510006

//BE receive DB command
#define COMMAND_ID_BE_DB_SEND_DICOM_SERIES 610000
#define COMMAND_ID_BE_DB_SEND_PREPROCESS_MASK 610001
#define COMMAND_ID_BE_DB_SEND_AI_EVALUATION 610002
#define COMMAND_ID_BE_DB_SEND_ERROR 619999
#define COMMAND_ID_BE_DB_SEND_END 620000
#define COMMAND_ID_BE_DB_PACS_RETRIEVE_RESULT 610003
#define COMMAND_ID_BE_DB_PACS_FETCH_RESULT 610004

//AI receive DB operation
#define OPERATION_ID_AI_DB_REQUEST_AI_EVALUATION 630001

//DB receive AI operation
#define OPERATION_ID_DB_AI_EVALUATION_RESULT 630002
#define OPERATION_ID_DB_AI_READY 630003

//***************************************************************//

//***************************************************************//
// Model ID
#define MODEL_ID_ANNOTATION 910000
#define MODEL_ID_CROSSHAIR 910001
#define MODEL_ID_DBS_STATUS 910002
//***************************************************************//

MED_IMG_END_NAMESPACE

#endif