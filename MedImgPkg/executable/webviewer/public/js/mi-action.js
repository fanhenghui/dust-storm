/////////////////////////////////////////////////////////////
///Common action : rotate zoom pan windowing paging
/////////////////////////////////////////////////////////////
function ActionCommon(socketClient, cellID) {
    this.socketClient = socketClient;
    this.cellID = cellID;
    this.mouseClock = new Date().getTime();
};

ActionCommon.prototype.registerOpID = function(id) {
    this.opID = id;
}

ActionCommon.prototype.mouseDown = function(mouseBtn, mouseStatus, x, y, cell){
}

ActionCommon.prototype.mouseMove = function(mouseBtn, mouseStatus, x, y, preX, preY, cell){
    if(mouseStatus != BTN_DOWN) {
        return false;
    }

    var curClock = new Date().getTime();
    if (Math.abs(this.mouseClock - curClock) < MOUSE_MSG_INTERVAL) {
        return false;
    }
    //update mouse clock
    this.mouseClock = curClock;

    //send msg to BE
    if(!this.socketClient.protocRoot) {
        console.log('null protocbuf.');
        return;
    }
    var MsgMouse = this.socketClient.protocRoot.lookup('medical_imaging.MsgMouse');
    if(!MsgMouse) {
        console.log('get mouse message type failed.');
        return;
    }
    var msgMouse = MsgMouse.create({
      pre: {x: preX, y: preY},
      cur: {x: x, y: y},
      tag: 0
    });
    if(!msgMouse) {
        console.log('create mouse message failed.');
        return;
    }
    var msgBuffer = MsgMouse.encode(msgMouse).finish();
    this.socketClient.sendData(COMMAND_ID_FE_OPERATION, this.opID, this.cellID, msgBuffer.byteLength, msgBuffer);

    return true;
}

ActionCommon.prototype.mouseUp = function(mouseBtn, mouseStatus, x, y, cell){
}

/////////////////////////////////////////////////////////////
///Annotation action
/////////////////////////////////////////////////////////////
const ANNOTATION_ADD = 0;
const ANNOTATION_DELETE = 1;
const ANNOTATION_MODIFYING = 2;
const ANNOTATION_MODIFYCOMPLETED = 3;

function sendMSG(cellID, annoType, annoID, annoStatus, annoVis, para0, para1, para2, socketClient) {
    if(!socketClient.protocRoot) {
        console.log('null protocbuf.');
        return;
    }

    var MsgAnnotationUnit = socketClient.protocRoot.lookup('medical_imaging.MsgAnnotationUnit');
    if(!MsgAnnotationUnit) {
        console.log('get annotation unit message type failed.');
        return;
    }
    var msgAnnoUnit = MsgAnnotationUnit.create({
        type: annoType,
        id: annoID,
        status: annoStatus,
        visibility: annoVis,
        para0: para0,
        para1: para1,
        para2: para2
    });
    if(!msgAnnoUnit) {
        console.log('create annotation unit message failed.');
        return;
    }
    var msgBuffer = MsgAnnotationUnit.encode(msgAnnoUnit).finish();
    socketClient.sendData(COMMAND_ID_FE_OPERATION, ACTION_ID_MRP_ANNOTATION, cellID, msgBuffer.byteLength, msgBuffer);
}

function ActionAnnotation(socketClient, cellID) {
    this.socketClient = socketClient;
    this.mouseClock = new Date().getTime();
    this.cellId = cellID;
};

ActionAnnotation.prototype.mouseDown = function(mouseBtn, mouseStatus, x, y, cell){
    let annoID = cell.rois.length;
    //add a new ROI to operating cell
    cell.lastROI = new ROICircle(cell.rois.length, cell.svg, x, y, annoID);

    //bind drag callback
    cell.lastROI.dragingCallback = (function(cx, cy, r, key) {
        sendMSG(this.cellID, 0, key, ANNOTATION_MODIFYING, true, cx, cy, r, this.socketClient);
        return true;
    }).bind(this);

    cell.lastROI.dragEndCallback = (function(cx, cy, r, key) {
        sendMSG(this.cellID, 0, key, ANNOTATION_MODIFYCOMPLETED, true, cx, cy, r, this.socketClient);
        return true;
    }).bind(this);

    //send msg to BE
    let cx = cell.lastROI.cx;
    let cy = cell.lastROI.cy;
    let r = cell.lastROI.r;
    sendMSG(this.cellID, 0, annoID, ANNOTATION_ADD, true, cx, cy, r, this.socketClient);
}

ActionAnnotation.prototype.mouseMove = function(mouseBtn, mouseStatus, x, y, preX, preY, cell){
    if(mouseStatus != BTN_DOWN) {
        return false;
    }

    //ROI shaping by mouse moving(directly)
    cell.lastROI.creating(x, y);

    var curClock = new Date().getTime();
    if (Math.abs(this.mouseClock - curClock) < MOUSE_MSG_INTERVAL) {
        return false;
    }

    //send msg to BE
    let annoID = cell.rois.length;
    let cx = cell.lastROI.cx;
    let cy = cell.lastROI.cy;
    let r = cell.lastROI.r;
    sendMSG(this.cellID, 0, annoID, ANNOTATION_MODIFYING, true, cx, cy, r, this.socketClient);

    return true;
}

ActionAnnotation.prototype.mouseUp = function(mouseBtn, mouseStatus, x, y, cell){
    //add shaped roi to roi arrays
    cell.rois.push(cell.lastROI);
    cell.lastROI = null;

    //send msg to BE
    let roi = cell.rois[cell.rois.length-1];
    let annoID = cell.rois.length-1;
    let cx = roi.cx;
    let cy = roi.cy;
    let r = roi.r;
    sendMSG(this.cellID, 0, annoID, ANNOTATION_MODIFYCOMPLETED, true, cx, cy, r, this.socketClient);
}
