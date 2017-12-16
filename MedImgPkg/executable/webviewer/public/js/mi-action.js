/////////////////////////////////////////////////////////////
///Common action : rotate zoom pan windowing paging
/////////////////////////////////////////////////////////////
function ActionCommon(socketClient, cellID) {
    this.socketClient = socketClient;
    this.cellID = cellID;
    this.mouseClock = new Date().getTime();
    this.opID;
    this.leftOpBtnID;
    this.rightOpBtnID;
    this.midOpBtnID;
};

function sendDownsampleMSG(cellID, flag, socketClient) {
    if(!socketClient.protocRoot) {
        console.log('null protocbuf.');
        return;
    }

    let MsgFlag = socketClient.protocRoot.lookup('medical_imaging.MsgFlag');
    if(!MsgFlag) {
        console.log('get flag message type failed.');
        return;
    }
    let msgFlag = MsgFlag.create({
        flag: flag
    });
    if(!msgFlag) {
        console.log('create flag message failed.');
        return;
    }
    let msgBuffer = MsgFlag.encode(msgFlag).finish();
    socketClient.sendData(COMMAND_ID_BE_FE_OPERATION, OPERATION_ID_BE_FE_DOWNSAMPLE, cellID, msgBuffer.byteLength, msgBuffer);
}

ActionCommon.prototype.registerOpID = function(left, right, mid, hold) {
    this.leftOpBtnID = left;
    this.rightOpBtnID = right;
    this.midOpBtnID = mid;
    this.hodlOpBtnID = hold;
}

ActionCommon.prototype.mouseDown = function(mouseBtn, mouseStatus, x, y, cell){
    sendDownsampleMSG(this.cellID, true, this.socketClient);
}

ActionCommon.prototype.mouseMove = function(mouseBtn, mouseStatus, x, y, preX, preY, cell){
    if(mouseStatus != BTN_DOWN) {
        return false;
    }

    let curClock = new Date().getTime();
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
    let MsgMouse = this.socketClient.protocRoot.lookup('medical_imaging.MsgMouse');
    if(!MsgMouse) {
        console.log('get mouse message type failed.');
        return;
    }
    let msgMouse = MsgMouse.create({
      pre: {x: preX, y: preY},
      cur: {x: x, y: y},
      tag: 0
    });
    if(!msgMouse) {
        console.log('create mouse message failed.');
        return;
    }
    let msgBuffer = MsgMouse.encode(msgMouse).finish();
    let opID = this.leftOpBtnID;
    if (mouseBtn == BTN_LEFT) {
        opID = this.leftOpBtnID;
    } else if (mouseBtn == BTN_RIGHT) {
        opID = this.rightOpBtnID;
    } else if (mouseBtn == BTN_MIDDLE){
        opID = this.midOpBtnID;
    } else if (mouseBtn == (parseInt(BTN_LEFT) | parseInt(BTN_RIGHT)) ) {
        opID = this.hodlOpBtnID;
    } else {
        return;
    }

    this.socketClient.sendData(COMMAND_ID_BE_FE_OPERATION, opID, this.cellID, msgBuffer.byteLength, msgBuffer);

    return true;
}

ActionCommon.prototype.mouseUp = function(mouseBtn, mouseStatus, x, y, cell) {
    sendDownsampleMSG(this.cellID, false, this.socketClient);
}

/////////////////////////////////////////////////////////////
///Annotation action
/////////////////////////////////////////////////////////////
const ANNOTATION_ADD = 0;
const ANNOTATION_DELETE = 1;
const ANNOTATION_MODIFYING = 2;
const ANNOTATION_MODIFYCOMPLETED = 3;
const ANNOTATION_FOCUS = 4;

function sendAnnotationMSG(cellID, annoType, annoID, annoStatus, annoVis, para0, para1, para2, probability, socketClient) {
    if(!socketClient.protocRoot) {
        console.log('null protocbuf.');
        return;
    }

    let MsgAnnotationUnit = socketClient.protocRoot.lookup('medical_imaging.MsgAnnotationUnit');
    if(!MsgAnnotationUnit) {
        console.log('get annotation unit message type failed.');
        return;
    }
    let msgAnnoUnit = MsgAnnotationUnit.create({
        type: annoType,
        id: annoID,
        status: annoStatus,
        visibility: annoVis,
        para0: para0,
        para1: para1,
        para2: para2,
        probability: probability, 
    });
    if(!msgAnnoUnit) {
        console.log('create annotation unit message failed.');
        return;
    }
    let msgBuffer = MsgAnnotationUnit.encode(msgAnnoUnit).finish();
    socketClient.sendData(COMMAND_ID_BE_FE_OPERATION, ACTION_ID_MRP_ANNOTATION, cellID, msgBuffer.byteLength, msgBuffer);
}

function ActionAnnotation(socketClient, cellID) {
    this.socketClient = socketClient;
    this.mouseClock = new Date().getTime();
    this.cellID = cellID;
    this.upCallback = null;
};

ActionAnnotation.prototype.createROICircle = function(id, svg, cx, cy, r, visibility, probability, contentStr) {
    let roi = new ROICircle(id, svg, cx, cy, r);
    roi.visible(visibility);
    roi.setCtrlRadius(0.0);
    roi.addAnnotationLabel(contentStr); // temporily placed here
    roi.probability = probability;//add a new attribute
    //bind drag callback
    roi.dragingCallback = (function(cx, cy, r, key) {
        sendAnnotationMSG(this.cellID, 0, key, ANNOTATION_MODIFYING, true, cx, cy, r, probability, this.socketClient);
        return true;
    }).bind(this);

    roi.dragEndCallback = (function(cx, cy, r, key) {
        sendAnnotationMSG(this.cellID, 0, key, ANNOTATION_MODIFYCOMPLETED, true, cx, cy, r, probability, this.socketClient);
        return true;
    }).bind(this);

    return roi;
}

ActionAnnotation.prototype.mouseDown = function(mouseBtn, mouseStatus, x, y, cell){
    let annoID = 'eva-' + new Date().getTime() + '-' + cell.rois.length;

    //add a new ROI to operating cell
    let probability = 1.0;//created by user, default probability is 1.0
    cell.lastROI = this.createROICircle(annoID, cell.svg, x, y, 0, probability, true);
    
    //send msg to BE
    let cx = cell.lastROI.cx;
    let cy = cell.lastROI.cy;
    let r = cell.lastROI.r;
    sendAnnotationMSG(this.cellID, 0, annoID, ANNOTATION_ADD, true, cx, cy, r, probability, this.socketClient);
}

ActionAnnotation.prototype.mouseMove = function(mouseBtn, mouseStatus, x, y, preX, preY, cell) {
    if(mouseStatus != BTN_DOWN) {
        return false;
    }

    //ROI shaping by mouse moving(directly)
    cell.lastROI.creating(x, y);

    let curClock = new Date().getTime();
    if (Math.abs(this.mouseClock - curClock) < MOUSE_MSG_INTERVAL) {
        return false;
    }
    //update mouse clock
    this.mouseClock = curClock;

    //send msg to BE
    let annoID = cell.lastROI.key;
    let cx = cell.lastROI.cx;
    let cy = cell.lastROI.cy;
    let r = cell.lastROI.r;
    let probability = cell.lastROI.probability;
    sendAnnotationMSG(this.cellID, 0, annoID, ANNOTATION_MODIFYING, true, cx, cy, r, probability, this.socketClient);

    return true;
}

ActionAnnotation.prototype.mouseUp = function(mouseBtn, mouseStatus, x, y, cell) {
    if (null == cell.lastROI) {
        return;
    }

    //add shaped roi to roi arrays
    cell.rois.push(cell.lastROI);
    cell.lastROI = null;

    //send msg to BE
    let roi = cell.rois[cell.rois.length-1];
    let annoID = roi.key;
    let cx = roi.cx;
    let cy = roi.cy;
    let r = roi.r;
    let probability = roi.probability;
    sendAnnotationMSG(this.cellID, 0, annoID, ANNOTATION_MODIFYCOMPLETED, true, cx, cy, r, probability, this.socketClient);

    if(this.upCallback) {
        this.upCallback();
    }
}

