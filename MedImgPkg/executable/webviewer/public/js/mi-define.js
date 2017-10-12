// FE to BE command ID
const COMMAND_ID_FE_SHUT_DOWN = 120000;
const COMMAND_ID_FE_READY = 120001;
const COMMAND_ID_FE_OPERATION = 120002;
const COMMAND_ID_FE_MPR_PLAY = 120003;
const COMMAND_ID_FE_VR_PLAY = 120004;
const COMMAND_ID_FE_SEARCH_WORKLIST = 120005;

// BE to FE command ID
const COMMAND_ID_BE_HEARTBEAT = 269999;
const COMMAND_ID_BE_READY = 270000;
const COMMAND_ID_BE_SEND_IMAGE = 270001;
const COMMAND_ID_BE_SEND_WORKLIST = 270002;
const COMMAND_ID_BE_SEND_ANNOTATION_LIST = 270003;
const COMMAND_ID_BE_SEND_NONE_IMAGE = 270004;

//cell operation ID
const OPERATION_ID_INIT = 310000;
const OPERATION_ID_MPR_PAGING = 310001;
const OPERATION_ID_PAN = 310002;
const OPERATION_ID_ZOOM = 310003;
const OPERATION_ID_ROTATE = 310004;
const OPERATION_ID_WINDOWING = 310005;
const OPERATION_ID_RESIZE = 310006;
const OPERATION_ID_ANNOTATION = 310007;
const OPERATION_ID_MPR_MASK_OVERLAY = 310008;
const OPERATION_ID_SWITCH_PRESET_WINDOWING = 310009

//mouse action
const ACTION_ID_NONE = 0;
const ACTION_ID_MPR_PAGING = OPERATION_ID_MPR_PAGING; 
const ACTION_ID_ZOOM = OPERATION_ID_ZOOM;
const ACTION_ID_PAN = OPERATION_ID_PAN;
const ACTION_ID_ROTATE = OPERATION_ID_ROTATE;
const ACTION_ID_WINDOWING = OPERATION_ID_WINDOWING;
const ACTION_ID_MRP_ANNOTATION = OPERATION_ID_ANNOTATION;

const PROTOBUF_BE_FE = './data/mi_message.proto';

//mouse button type
const BTN_NONE = -1;
const BTN_LEFT = 0;
const BTN_MIDDLE = 1;
const BTN_RIGHT = 2;

//mouse button status
const BTN_DOWN = 0;
const BTN_UP = 1;

//mouse message interval
const MOUSE_MSG_INTERVAL = 10;