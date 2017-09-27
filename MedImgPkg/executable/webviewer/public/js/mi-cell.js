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

    //svg(d3.js) based graphic
    this.noneImgBuf = null;

    //mouse event
    this.mouseAction = ACTION_ID_NONE;
    this.mouseBtn = BTN_NONE;
    this.mouseStatus = BTN_UP;
    this.mousePre = {x:0,y:0};
    this.mouseClock = new Date().getTime();
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

Cell.prototype.handleNongImgBuffer = function (tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader) {
    if (withHeader) {
        noneImgBuf = new ArrayBuffer(dataLen + restDataLen);
    }
    var dstview = new Uint8Array(noneImgBuf);
    var srcview = new Uint8Array(tcpBuffer, bufferOffset, dataLen);
    var cpSrc = noneImgBuf.byteLength - (dataLen + restDataLen);
    for (var i = cpSrc; i < dataLen; i++) {
        dstview[i] = srcview[i];
    }

    // sucessfully receive all the incoming text
    if (restDataLen <= 0) {
        console.log('receive Nong Img Buffer.');
        var MsgNoneImgCollection = socketClient.protocRoot.lookup('medical_imaging.MsgNoneImgCollection');
        if (!MsgNoneImgCollection) {
            console.log('get MsgNoneImgCollection type failed.');
        }

        //decode the byte array with protobuffer
        var noneImgBufView = new Uint8Array(noneImgBuf);
        var receivedMsg = MsgNoneImgCollection.decode(noneImgBufView);
        if (receivedMsg.cornerInfos) // contains the 2nd optional array
        {
            var txt = receivedMsg.cornerInfos.infos;
            var corners = txt.split('\n');

            // for each corner
            // 'length-1': avoid the last empty string "" after '\n'
            xpos = [4, 4, this.canvas.width-4, this.canvas.width-4];
            for (i = 0; i < (corners.length - 1); ++i) {
                var multiLine = corners[i].split('|');
                // j starts from 2 (instead of 1) to avoid a redudent "" after the first '|'
                var inforArray = [];
                for (j = 2; j < multiLine.length; ++j) {
                    var info = multiLine[j].split(':');
                    inforArray.push(info[1]);
                }
                
                var svgText = d3.select('#' + this.svg.id).select('#' + multiLine[0])
                var txtspacing = (i % 2 == 0 ? 1 : -1) * parseFloat(svgText.attr('font-size'));
                var tspans = svgText.selectAll('tspan').data(inforArray);
                tspans.enter()
                    .append('tspan')
                    .attr('x', xpos[i])
                    .attr('dy', function (d, i) {
                        return i ? (1.2 * txtspacing) : txtspacing
                    })
                    .text(function (d, i) {
                        return d;
                    });
                tspans.exit().remove();
            }
        }
    }
}

Cell.prototype.resize = function(width, height) {
    //canvas&svg resize
    //send msg to notigy BE resize will be call in main
    this.canvas.width = width;
    this.canvas.height = height;
    var top = this.canvas.offsetTop;
    var left = this.canvas.offsetLeft;
    var viewBox = left.toString() + ' ' + top + ' ' + width + ' ' + height; 
    this.svg.setAttribute('viewBox', viewBox);
    this.svg.setAttribute('width', width);
    this.svg.setAttribute('height', height);
    this.svg.setAttribute('x', left);
    this.svg.setAttribute('y', top);
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
    //update mouse clock
    this.mouseClock = curClock;

    //create mouse msg and send to BE
    if(!socketClient.protocRoot) {
        console.log('null protocbuf.');
        return;
    }
    var MsgMouse = socketClient.protocRoot.lookup('medical_imaging.MsgMouse');
    if(!MsgMouse) {
        console.log('get mouse message type failed.');
        return;
    }
    var msgMouse = MsgMouse.create({
      pre: {x: this.mousePre.x, y: this.mousePre.y},
      cur: {x: curPos.x, y: curPos.y},
      tag: 0
    });
    if(!msgMouse) {
        console.log('create mouse message failed.');
        return;
    }
    var msgBuffer = MsgMouse.encode(msgMouse).finish();
    socketClient.sendData(COMMAND_ID_FE_OPERATION, this.mouseAction, this.cellID, msgBuffer.byteLength, msgBuffer);

    //reset previous position
    this.mousePre.x = curPos.x;
    this.mousePre.y = curPos.y;
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

        // add texts at four corners
        var width = this.canvas.width;
        var height = this.canvas.height;
        d3.select('#' + this.svg.id)
            .selectAll('text')
            .data(["LT", "LB", "RT", "RB"])
            .enter()
            .append("text")
            .attr('font-family', 'monospace')
            .attr('fill', '#dcdcdc')
            .attr('alignment-baseline', function (d, i) {
                return (i % 2 == 0) ? 'hanging' : 'baseline';
            })
            .attr('text-anchor', function (d, i) {
                return (i < 2) ? "start" : 'end';
            })
            .attr('x', function (d, i) {
                return (i < 2) ? "4px" : width - 4;
            })
            .attr('y', function (d, i) {
                return (i % 2 == 0) ? "4px" : height - 4;
            })
            .attr('id', function (d) {
                return d;
            })
            .attr('font-size', "15px");
    }
}