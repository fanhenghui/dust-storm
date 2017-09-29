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

const TEXT_MARGIN = 4;
//primitive
const ADD_PRIMITIVE = 0;
const DELETE_PRIMITIVE = 1;
const MODIFYING_PRIMITIVE = 2;
const DONE_PRIMITIVE = 3;

const LBN_NOTHING_PRIMITIVE = -1;
const LBN_CREATING_PRIMITIVE = 0;
const LBN_MODIFYING_PRIMITIVE = 1;

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

    //primitiveID
    this.primitiveArray = [];
    this.selectedPrimitive = null;
    this.primitiveProcess = LBN_NOTHING_PRIMITIVE; // -1: none; 0: creating; 1: modifying
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
    var dstRecord = new Uint8Array(noneImgBuf);
    var srcRecord = new Uint8Array(tcpBuffer, bufferOffset, dataLen);
    var cpSrc = noneImgBuf.byteLength - (dataLen + restDataLen);
    for (var i = cpSrc; i < dataLen; i++) {
        dstRecord[i] = srcRecord[i];
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
        if (receivedMsg.cornerInfos) {
            var txt = receivedMsg.cornerInfos.infos;//MSG Format "LT|1:patientName|2:patientID\nLB....\nRT|....\nRB|....\n"
            var corners = txt.split('\n');

            // for each corner
            // 'length-1': avoid the last empty string "" after '\n'
            for (var i = 0; i < (corners.length - 1); ++i) {
                var oneCornerInfo = corners[i].split('|');
                let cornerTag = oneCornerInfo[0];
                var infoContext = [];
                for (var j = 1; j < oneCornerInfo.length; ++j) {
                    var info = oneCornerInfo[j].split(',');
                    infoContext.push({pos:info[0], context:info[1]});
                }
                
                var svgText = d3.select(this.svg).select('#' + cornerTag);
                switch(cornerTag) {
                    case 'LT':
                    var xpos = TEXT_MARGIN;
                    var txtspacing = parseFloat(svgText.attr('font-size'));
                    var texorigin = 1.2*txtspacing;
                    break;
                    case 'RT':
                    var xpos = this.canvas.width - TEXT_MARGIN;
                    var txtspacing = parseFloat(svgText.attr('font-size'));
                    var texorigin = 1.2*txtspacing;
                    break;
                    
                    case 'LB':
                    var xpos = TEXT_MARGIN;
                    var txtspacing = -parseFloat(svgText.attr('font-size'));
                    var texorigin = this.canvas.height + 0.2*txtspacing;
                    break;
                    case 'RB':
                    var xpos = this.canvas.width - TEXT_MARGIN;
                    var txtspacing = -parseFloat(svgText.attr('font-size'));
                    var texorigin = this.canvas.height + 0.2*txtspacing;
                    break;
                }                

                var tspans = svgText.selectAll('tspan');
                if(tspans.empty())
                {
                    tspans.data(infoContext).enter()
                    .append('tspan')
                    .attr('x', xpos)
                    .attr('y', function (d) {
                        return texorigin + (d.pos)*txtspacing;
                    })
                    .attr('id', function (d) {
                        return cornerTag + '-' + d.pos;
                    })
                    .text(function (d, i) {
                        return d.context;
                    });
                } else {
                    tspans.data(infoContext, function(d) {//change data[d.pos]
                        return d.pos;
                    })
                    .text(function (d, i) {
                        return d.context;
                    });
                }
            }
        } else if (receivedMsg.annotations) {
            
        }
    }
}

Cell.prototype.resize = function(width, height) {
    //canvas&svg resize
    //send msg to notigy BE resize will be call in main
    this.canvas.width = Math.floor(width);
    this.canvas.height = Math.floor(height);
    var top = this.canvas.offsetTop;
    var left = this.canvas.offsetLeft;
    var viewBox = left.toString() + ' ' + top + ' ' + Math.floor(width) + ' ' + Math.floor(height); 
    this.svg.setAttribute('viewBox', viewBox);
    this.svg.setAttribute('width', Math.floor(width));
    this.svg.setAttribute('height', Math.floor(height));
    this.svg.setAttribute('x', left);
    this.svg.setAttribute('y', top);

    // resize the txts located at 4 corners
    d3.select(this.svg)
        .selectAll('text')
        .attr('x', function(d, i) { return (i < 2) ? TEXT_MARGIN : width - TEXT_MARGIN; })
        .attr('y', function(d, i) { return (i % 2 == 0) ? TEXT_MARGIN : height - TEXT_MARGIN; })
        .each(function(d, i) {
          d3.select(this).selectAll('tspan').attr(
              'x', function(datum, j) { return (i < 2) ? TEXT_MARGIN : width - TEXT_MARGIN; })
        });
}

Cell.prototype.mouseDown = function(event) {
    this.mouseStatus = BTN_DOWN;
    this.mouseBtn = event.button;

    var x = event.clientX - this.svg.getBoundingClientRect().left;
    var y = event.clientY - this.svg.getBoundingClientRect().top;
    this.mousePre.x = x;
    this.mousePre.y = y;

    // send a msg to notify BE we are adding an circle
    if(this.mouseAction == ACTION_ID_MRP_ANNOTATION && this.mouseBtn == BTN_LEFT && !this.selectedPrimitive)
    {
        this.processAnnotationAction({'x':x,'y':y}, ADD_PRIMITIVE); //0:add
    }
}

Cell.prototype.mouseMove = function (event) {
    if (this.mouseStatus != BTN_DOWN) {
        return;
    }

    var curX = event.clientX - this.svg.getBoundingClientRect().left;
    var curY = event.clientY - this.svg.getBoundingClientRect().top;
    if (this.mouseAction == ACTION_ID_MRP_ANNOTATION) {
        this.processAnnotationAction({'x': curX,'y': curY}, MODIFYING_PRIMITIVE); //2:modify
    } else {
        this.processMouseAction({x: curX,y: curY});
    }
    //reset previous position
    this.mousePre.x = curX;
    this.mousePre.y = curY;
}

Cell.prototype.processAnnotationAction = function (curPos, status) {
    if (this.svg == null) {
        return;
    }

    var MsgAnnotationUnit = socketClient.protocRoot.lookup('medical_imaging.MsgAnnotationUnit');
    var msgAnnoUnit = MsgAnnotationUnit.create();
    msgAnnoUnit.type = 0;
    msgAnnoUnit.status = status;
    msgAnnoUnit.visibility = true;

    if (status == ADD_PRIMITIVE) {
        // visual representation @ FE
        var newCircle = d3.select(this.svg).append("circle")
            .attr("cx", curPos.x)
            .attr("cy", curPos.y)
            .attr("r", 1)
            .style("fill-opacity", 0.0) //热点是整个圆
            //.style("fill", 'none')//热点是圆圈
            .style("stroke", "red")
            .style("stroke-opacity", 0.8)
            .style("stroke-width", 2)
            .style("cursor", "move")
            .on("contextmenu", function (data, index) {
                d3.event.preventDefault();
                return false;
            }).on("mousedown", function (event) {
            //TODO: mark the currently selected primitive : d3.select(event.target).attr('id');
            this.mouseAction = ACTION_ID_MRP_ANNOTATION;
            this.mouseStatus = BTN_DOWN;
            this.selectedPrimitive = d3.select(event.target);
            // document.onmousemove = Cell.prototype.mouseMove;
            // document.onmouseup = Cell.prototype.mouseUp;
        }.bind(this));

        
        this.primitiveArray.push(newCircle._groups[0][0]);
        this.selectedPrimitive = newCircle;
        msgAnnoUnit.id = this.primitiveArray.indexOf(this.selectedPrimitive._groups[0][0]);

        msgAnnoUnit.para0 = curPos.x;
        msgAnnoUnit.para1 = curPos.y;
        msgAnnoUnit.para2 = 1;
        var msgBuffer = MsgAnnotationUnit.encode(msgAnnoUnit).finish();
        socketClient.sendData(COMMAND_ID_FE_OPERATION, OPERATION_ID_ANNOTATION, 0, msgBuffer.byteLength, msgBuffer);
    } else if (status == MODIFYING_PRIMITIVE) {
        //prevent mouse msg too dense
        var curClock = new Date().getTime();
        if (Math.abs(this.mouseClock - curClock) < MOUSE_MSG_INTERVAL) {
            return;
        }
        //update mouse clock
        this.mouseClock = curClock;
        msgAnnoUnit.id = this.primitiveArray.indexOf(this.selectedPrimitive._groups[0][0]);
        // Check left (change radius) or right button (change position) is pressed
        if (this.mouseBtn == BTN_LEFT) {
            var cx = this.selectedPrimitive.attr('cx');
            var cy = this.selectedPrimitive.attr('cy');
            var r = Math.sqrt((curPos.x - cx) * (curPos.x - cx) + (curPos.y - cy) * (curPos.y - cy));
            this.selectedPrimitive.attr('r', r);
        } else if (this.mouseBtn == BTN_MIDDLE) {
            this.selectedPrimitive.attr('cx', curPos.x);
            this.selectedPrimitive.attr('cy', curPos.y);
        }
        msgAnnoUnit.para0 = this.selectedPrimitive.attr('cx');
        msgAnnoUnit.para1 = this.selectedPrimitive.attr('cy');
        msgAnnoUnit.para2 = this.selectedPrimitive.attr('r');
        var msgBuffer = MsgAnnotationUnit.encode(msgAnnoUnit).finish();
        socketClient.sendData(COMMAND_ID_FE_OPERATION, OPERATION_ID_ANNOTATION, 0, msgBuffer.byteLength, msgBuffer);

    } else if (status == DONE_PRIMITIVE) {
        // underlying data communication with the BE
        msgAnnoUnit.id = this.primitiveArray.indexOf(this.selectedPrimitive._groups[0][0]);
        msgAnnoUnit.para0 = this.selectedPrimitive.attr('cx');
        msgAnnoUnit.para1 = this.selectedPrimitive.attr('cy');
        msgAnnoUnit.para2 = this.selectedPrimitive.attr('r');
        var msgBuffer = MsgAnnotationUnit.encode(msgAnnoUnit).finish();
        socketClient.sendData(COMMAND_ID_FE_OPERATION, OPERATION_ID_ANNOTATION, 0, msgBuffer.byteLength, msgBuffer);
    } else if (status == DELETE_PRIMITIVE) // delete
    {
        msgAnnoUnit.id = this.primitiveArray.indexOf(this.selectedPrimitive._groups[0][0]);
        this.primitiveArray = this.primitiveArray.splice(msgAnnoUnit.id,1);
        this.selectedPrimitive.remove();
        msgAnnoUnit.visibility = false;
        msgAnnoUnit.para0 = 0;
        msgAnnoUnit.para1 = 0;
        msgAnnoUnit.para2 = 0;
        var msgBuffer = MsgAnnotationUnit.encode(msgAnnoUnit).finish();
        socketClient.sendData(COMMAND_ID_FE_OPERATION, OPERATION_ID_ANNOTATION, 0, msgBuffer.byteLength, msgBuffer);
    } else {;
    }
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
}

Cell.prototype.mouseUp = function(event) {
    this.mouseBtn = BTN_NONE;
    this.mouseStatus = BTN_UP;
    this.mousePre.x = 0;
    this.mousePre.y = 0;

    // send a msg to notify BE we are done with adding an circle
    if (this.mouseAction == ACTION_ID_MRP_ANNOTATION) {
        this.processAnnotationAction({x: 0,y: 0}, DONE_PRIMITIVE); //3:done
        this.selectedPrimitive = null;
        // document.onmousemove = null;
        // document.onmouseup = null;
    }
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
        d3.select(this.svg)
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
                return (i < 2) ? "4px" : width - TEXT_MARGIN;
            })
            .attr('y', function (d, i) {
                return (i % 2 == 0) ? "4px" : height - TEXT_MARGIN;
            })
            .attr('id', function (d) {
                return d;
            })
            .attr('font-size', "15px")
            .attr('class', 'no-select-text');
    }
}