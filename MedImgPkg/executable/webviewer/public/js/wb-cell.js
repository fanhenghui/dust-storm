//mouse button type
const BTN_NONE = -1;
const BTN_LEFT = 0;
const BTN_MIDDLE = 1;
const BTN_RIGHT = 2;

//mouse button status
const BTN_DOWN = 0;
const BTN_UP = 1;

//mouse interval
const MOUSE_MSG_INTERVAL = 10;

function Cell(cellName, cellID, canvas, svg, socketClient) {
    this.cellName = cellName;
    this.cellID = cellID;
    
    //containee canvas& svg
    this.canvas = canvas;
    this.svg = svg;

    //web-socket-client
    this.socketClient = socketClient;

    //canvas JPEG based draw
    this.jpegStr = '';
    this.jpegImg = new Image();

    //mouse event
    this.mouseAction = ACTION_ID_NONE;
    this.mouseBtn = BTN_NONE;
    this.mouseStatus = BTN_UP;
    this.mousePre = {x:0,y:0};
    this.mouseClock = new Date().getTime();
    
    //register event linsener

}

function refreshCanvas(canvas, img) {
    canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
}

Cell.prototype.handleJpegBuffer = function(tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader) {
    if (withHeader) {
        this.jpegStr = '';
    } 
    var imgBuffer = new Uint8Array(tcpBuffer, bufferOffset, dataLen);
    this.jpegStr += String.fromCharCode.apply(null, imgBuffer);
    if(restDataLen <= 0) {
        this.jpegImg.src =  'data:image/jpg;base64,' + btoa(this.jpegStr);
        loadImg = (function(){
            refreshCanvas(this.canvas, this.jpegImg);
        }).bind(this);
        this.jpegImg.onload = loadImg;
    }
}

Cell.prototype.resize = function(width, height) {
    //canvas resize
    this.canvas.width = width;
    this.canvas.height = height;

    // this.svg.width = width;
    // this.svg.height = height;
    var tm = "#"+this.svg.id;
    d3.select(tm)
    .attr('width', width)
    .attr('height', height);
    //TODO svg resize

    //send msg to notigy BE resize will be call outside
}

Cell.prototype.mouseDown = function(event) {
    this.mouseStatus = BTN_DOWN;
    this.mouseBtn = event.button;

    var x = event.clientX - event.toElement.getBoundingClientRect().left;
    var y = event.clientY - event.toElement.getBoundingClientRect().top;
    this.mousePre.x = x;
    this.mousePre.y = y;
}

Cell.prototype.mouseMove = function(event) {
    if (this.mouseBtn != BTN_DOWN) {
        return;
    }

    var curX = event.clientX - event.toElement.getBoundingClientRect().left;
    var curY = event.clientY - event.toElement.getBoundingClientRect().top;
    this.processMouseAction({x:curX,y:curY});
}

Cell.prototype.processMouseAction = function(curPos) {
    //prevent mouse msg too dense
    var curClock = new Date().getTime();
    if (Math.abs(this.mouseClock - curClock) < MOUSE_MSG_INTERVAL) {
        return;
    }
    this.mouseClock = curClock;

    //create mouse msg and send to BE
    if(!socketClient.protocRoot) {
        //TODO log
        return;
    }
    var MsgMouse = socketClient.protocRoot.lookup('medical_imaging.MsgMouse');
    if(!MsgMouse) {
        //TOOD log
        return;
    }
    var msgMouse = MsgMouse.create({
      pre: {x: this.mousePre.x, y: this.mousePre.y},
      cur: {x: curPos.x, y: curPos.y},
      tag: 0
    });
    if(!msgMouse) {
        //TOOD log
        return;
    }
    var msgBuffer = MsgMouse.encode(msgMouse).finish();
    socketClient.sendData(COMMAND_ID_FE_OPERATION, this.mouseAction, this.cellID, msgBuffer.byteLength, msgBuffer);
}

Cell.prototype.mouseUp = function(event) {
    this.mouseBtn = BTN_NONE;
    this.mouseStatus = BTN_UP;
    this.mousePre.x = 0;
    this.mousePre.y = 0;
}

Cell.prototype.prepare = function() {
    if(this.svg != null) {
        var mouseDown_ = (function(event) {
            this.mouseDown(event);
        }).bind(this);
        this.svg.addEventListener('mousedown', mouseDown_);

        var mouseMove_ = (function(event) {
            this.mouseMove(event);
        }).bind(this);
        this.svg.addEventListener('mousemove', mouseMove_);

        var mouseUp_ = (function(event) {
            this.mouseUp(event);
        }).bind(this);
        this.svg.addEventListener('mouseup', mouseUp_);
        return true;
    } else {
        return false;
    }
}