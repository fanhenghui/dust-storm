(function() {
    // FE to BE
    COMMAND_ID_FE_SHUT_DOWN = 120000;
    COMMAND_ID_FE_READY = 120001;
    COMMAND_ID_FE_OPERATION = 120002;
    COMMAND_ID_FE_MPR_PLAY = 120003;
    COMMAND_ID_FE_VR_PLAY = 120004;

    // BE to FE
    COMMAND_ID_BE_SEND_IMAGE = 270001;
    COMMAND_ID_BE_READY = 270000;

    // FE to BE Operation ID
    OPERATION_ID_INIT = 310000;
    OPERATION_ID_MPR_PAGING = 310001;
    OPERATION_ID_PAN = 310002;
    OPERATION_ID_ZOOM = 310003;
    OPERATION_ID_ROTATE = 310004;

    // init
    cellCanvas = [
        document.getElementById('canvas0'), document.getElementById('canvas1'),
        document.getElementById('canvas2'), document.getElementById('canvas3')
    ];
    cellJpeg = ['', '', '', ''];
    cellImage = [new Image(), new Image(), new Image(), new Image()];

    // init canvas size
    console.log('w:' + window.innerWidth);
    console.log('h:' + window.innerHeight);

    cellContainerWidth = document.getElementById('cell-container').offsetWidth;
    cellContainerHeight = document.getElementById('cell-container').offsetHeight;
    navigatorHeight = document.getElementById('navigator-div').offsetHeight;

    console.log('cells w:' + cellContainerWidth);
    console.log('cells h:' + cellContainerHeight);

    for (var index = 0; index < cellCanvas.length; index++) {
        cellCanvas[index].width = (cellContainerWidth - 20) / 2;
        cellCanvas[index].height = (window.innerHeight - navigatorHeight - 40) / 2;
    }

    function refreshCanvas(cellID) {
        cellCanvas[cellID].getContext('2d').drawImage(
            cellImage[cellID], 0, 0, cellCanvas[cellID].width,
            cellCanvas[cellID].height);
    }

    function handleImage(cellID, tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader) {
        console.log('in render');
        if (withHeader) { //receive a new image
            cellJpeg[cellID] = '';
        }

        var imgBuffer = new Uint8Array(tcpBuffer, bufferOffset, dataLen);
        cellJpeg[cellID] += String.fromCharCode.apply(null, imgBuffer);

        if (restDataLen <= 0) {
            cellImage[cellID].src = 'data:image/jpg;base64,' + btoa(cellJpeg[cellID]);
            cellImage[cellID].onload = function() {
                //console.log('Image Onload');
                refreshCanvas(cellID);
            };
        }
    }

    function msgHandle(cmdID, cellID, opID, tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader) {
        switch (cmdID) {
            case COMMAND_ID_BE_SEND_IMAGE:
                0
                handleImage(cellID, tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader);
                break;
            case COMMAND_ID_BE_READY:
                window.FE.triggerOnBE();
                break;
            default:
                break;
        }
    }

    //TCP package related
    tcpPackageEnd = 0; //0 msg header 1 data for last msg 2 msg header for last msg header
    msgCmdID = 0;
    msgCellID = 0;
    msgOpID = 0;
    msgRestDataLen = 0;
    lastMsgHeader = new ArrayBuffer(32);
    lastMsgHeaderLen = 0;

    function processTCPMsg(tcpBuffer) {
        var tcpPackageLen = tcpBuffer.byteLength;

        if (tcpPackageEnd == 0) {
            if (tcpPackageLen < 32) {
                tcpPackageEnd = 2;
                lastMsgHeaderLen = tcpPackageLen;
                tcpBuffer.copy(lastMsgHeader, 0, 0);
                return;
            }
            var header = new Uint32Array(tcpBuffer, 0, 8);
            msgCmdID = header[2];
            msgCellID = header[3];
            msgOpID = header[4];
            var lastMsgDatalen = header[7];

            if (tcpPackageLen - 32 == lastMsgDatalen) { //completed one Msg
                msgHandle(msgCmdID, msgCellID, msgOpID, tcpBuffer, 32, lastMsgDatalen, 0, true);
            } else if (tcpPackageLen - 32 < lastMsgDatalen) { //not completed one Msg
                msgRestDataLen = lastMsgDatalen - (tcpPackageLen - 32);
                msgHandle(msgCmdID, msgCellID, msgOpID, tcpBuffer, 32, tcpPackageLen - 32, msgRestDataLen, true);
                tcpPackageEnd = 1;
            } else { //this buffer carry next Msg
                //process current one
                msgHandle(msgCmdID, msgCellID, msgOpID, tcpBuffer, 32, lastMsgDatalen, 0, true);
                //recursion process rest
                var tcpBufferSub = tcpBuffer.slice(lastMsgDatalen + 32);
                tcpPackageEnd = 0;
                processTCPMsg(tcpBufferSub);
            }
        } else if (tcpPackageEnd == 1) { //data for last msg
            if (tcpPackageLen - msgRestDataLen == 0) { //complete last msg
                msgRestDataLen = 0;
                msgHandle(msgCmdID, msgCellID, msgOpID, tcpBuffer, 0, tcpPackageLen, 0, false);
            } else if (tcpPackageLen - msgRestDataLen < 0) { //not complete data yet
                msgRestDataLen -= tcpPackageLen;
                tcpPackageEnd = 1;
                msgHandle(msgCmdID, msgCellID, msgOpID, tcpBuffer, 0, tcpPackageLen, msgRestDataLen, false);
            } else { //this buffer carry next Msg
                msgHandle(msgCmdID, msgCellID, msgOpID, tcpBuffer, 0, msgRestDataLen, 0, false);
                var tcpBufferSub2 = tcpBuffer.slice(msgRestDataLen);
                msgRestDataLen = 0;
                tcpPackageEnd = 0;
                processTCPMsg(tcpBufferSub2);
            }
        } else if (tcpPackageEnd == 2) { //msg header for last msg header
            var lastRestHeaderLen = 32 - lastMsgHeaderLen;
            if (tcpPackageLen < lastRestHeaderLen) { //msg header is not completed yet
                tcpPackageEnd = 2;
                tcpBuffer.copy(lastMsgHeader, 0, lastRestHeaderLen, tcpPackageLen);
                lastMsgHeaderLen += tcpPackageLen;
                return;
            } else { //msg header is completed
                tcpPackageEnd = 1;
                tcpBuffer.copy(lastMsgHeader, 0, lastRestHeaderLen, lastRestHeaderLen);
                var tcpBufferSub3 = tcpBuffer.slice(lastRestHeaderLen);

                var header2 = new Uint32Array(lastMsgHeader, 0, 8);
                msgCmdID = header2[2];
                msgCellID = header2[3];
                msgOpID = header2[4];
                msgRestDataLen = header2[7];

                tcpPackageEnd = 1;
                lastMsgHeaderLen = 0;
                processTCPMsg(tcpBufferSub3);
            }
        }
    }

    //////////////////////////////////////////////////////////////////
    //mouse event for test
    //ACTION ID
    ACTION_ID_ARROW = 400000;
    ACTION_ID_ZOOM = 400001;
    ACTION_ID_PAN = 400002;
    ACTION_ID_ROTATE = 400003;
    ACTION_ID_WINDOWING = 400004;


    BTN_NONE = -1;
    BTN_LEFT = 0;
    BTN_MIDDLE = 1;
    BTN_RIGHT = 2;

    BTN_DOWN = 0;
    BTN_UP = 1;

    //Btn status
    btnType = [BTN_NONE, BTN_NONE, BTN_NONE, BTN_NONE];
    btnStatus = [BTN_UP, BTN_UP, BTN_UP, BTN_UP];
    preMousePos = [{ x: 0, y: 0 }, { x: 0, y: 0 }, { x: 0, y: 0 }, { x: 0, y: 0 }];

    lastMouseMsg = { preX: 0, preY: 0, curX: 0, curY: 0 };

    function mouseMoveEvent(event) {
        var cellname = event.toElement.id;
        var cellid_s = cellname.slice(cellname.length - 1);
        var cellid = parseInt(cellid_s);
        if (btnStatus[cellid] != BTN_DOWN) {
            document.getElementById("test-info").innerText = "";
        } else {
            var button = event.button;
            var x = event.clientX - event.toElement.getBoundingClientRect().left;
            var y = event.clientY - event.toElement.getBoundingClientRect().top;
            document.getElementById("test-info").innerText = "move cell id : " + cellid.toString() +
                " " + x.toString() + " " + y.toString();

            if (preMousePos[cellid].x != x || preMousePos[cellid].y != y) {
                window.FE.rotate(cellid, { x: preMousePos[cellid].x, y: preMousePos[cellid].y }, { x: x, y: y });
                preMousePos[cellid].x = x;
                preMousePos[cellid].y = y;
            }
        }
    }

    function mouseDownEvent(event) {
        var cellname = event.toElement.id;
        var cellid_s = cellname.slice(cellname.length - 1);
        var cellid = parseInt(cellid_s);
        btnStatus[cellid] = BTN_DOWN;
        btnType[cellid] = event.button;

        var button = event.button;
        var x = event.clientX - event.toElement.getBoundingClientRect().left;
        var y = event.clientY - event.toElement.getBoundingClientRect().top;
        preMousePos[cellid].x = x;
        preMousePos[cellid].y = y;

        document.getElementById("test-info").innerText = "down" + cellname;
    }

    function mouseUpEvent(event) {
        var cellname = event.toElement.id;
        var cellid_s = cellname.slice(cellname.length - 1);
        var cellid = parseInt(cellid_s);
        btnStatus[cellid] = BTN_UP;
        btnType[cellid] = BTN_NONE;
        document.getElementById("test-info").innerText = "up" + cellname;
    }

    cellCanvas[0].addEventListener("mousemove", mouseMoveEvent);
    cellCanvas[0].addEventListener("mousedown", mouseDownEvent);
    cellCanvas[0].addEventListener("mouseup", mouseUpEvent);

    cellCanvas[1].addEventListener("mousemove", mouseMoveEvent);
    cellCanvas[1].addEventListener("mousedown", mouseDownEvent);
    cellCanvas[1].addEventListener("mouseup", mouseUpEvent);

    cellCanvas[2].addEventListener("mousemove", mouseMoveEvent);
    cellCanvas[2].addEventListener("mousedown", mouseDownEvent);
    cellCanvas[2].addEventListener("mouseup", mouseUpEvent);

    cellCanvas[3].addEventListener("mousemove", mouseMoveEvent);
    cellCanvas[3].addEventListener("mousedown", mouseDownEvent);
    cellCanvas[3].addEventListener("mouseup", mouseUpEvent);


    window.FE = {
            username: null,
            userid: null,
            socket: null,
            rotateTik: new Date().getTime(),

            resize: function() {
                console.log('resize');

                cellContainerWidth =
                    document.getElementById('cell-container').offsetWidth;
                cellContainerHeight =
                    document.getElementById('cell-container').offsetHeight;
                navigatorHeight = document.getElementById('navigator-div').offsetHeight;

                console.log('cells w:' + cellContainerWidth);
                console.log('cells h:' + cellContainerHeight);

                for (var index = 0; index < cellCanvas.length; index++) {
                    cellCanvas[index].width = (cellContainerWidth - 20) / 2;
                    cellCanvas[index].height =
                        (window.innerHeight - navigatorHeight - 40) / 2;
                }
            },

            genUID: function(username) {
                return username + new Date().getTime() + '' +
                    Math.floor(Math.random() * 173 + 511);
            },

            init: function(username) {
                //客户端根据时间和随机数生成uid,以用户名称可以重复，后续改为数据库的UID名称
                this.userid = this.genUID(username);
                this.username = username;

                //链接websocket服务器
                this.socket = io.connect('http://172.23.237.228:8000');

                //通知服务器有用户登录 TODO 这段逻辑应该在登录的时候做
                this.socket.emit('login', { userid: this.userid, username: this.username });

                //发送一段message
                this.socket.emit('message', {
                    userid: this.userid,
                    username: this.username,
                    content: 'first message'
                });


                this.socket.on('data', function(arraybuffer) {
                    console.log('receive data.');
                    processTCPMsg(arraybuffer);
                });

            },

            userLogOut: function() {
                //TODO login out 
                // this.socket.emit("logout",{userid:this.userid , username:this.username
                // , content:"last message"});
                // this.socket.emit("disconnect" , {userid:this.userid ,
                // username:this.username});
                location.reload();
            },

            userLogIn: function() {
                var username = document.getElementById('username').value;
                this.init(username);
            },

            paging: function() {
                var binding_func =
                    (function(err, root) {
                        this.socket = io.connect('http://172.23.237.226:8000');
                        if (err) {
                            console.log('load proto failed!');
                            throw err;
                        }
                        var MsgPaging = root.lookup('medical_imaging.MsgPaging');
                        var msgPaging = MsgPaging.create({ page: 1 });
                        var msgBuffer = MsgPaging.encode(msgPaging).finish();
                        var msgLength = msgBuffer.byteLength;
                        var cmdBuffer = new ArrayBuffer(32 + msgLength);

                        // header
                        var header = new Uint32Array(cmdBuffer, 0, 8);
                        header[0] = 0;
                        header[1] = 0;
                        header[2] = COMMAND_ID_FE_MPR_PLAY;
                        header[3] = 0;
                        header[4] = 0;
                        header[5] = 0;
                        header[6] = 0;
                        header[7] = msgLength;

                        // data
                        var srcBuffer = new Uint8Array(msgBuffer);
                        var dstBuffer = new Uint8Array(cmdBuffer, 8 * 4, msgLength);
                        for (var index = 0; index < msgLength; index++) {
                            dstBuffer[index] = srcBuffer[index];
                        }
                        console.log('emit paging message.');

                        this.socket.emit('data', {
                            userid: this.userid,
                            username: this.username,
                            content: cmdBuffer
                        });
                    }).bind(this);

                protobuf.load('./data/mi_message.proto', binding_func);
            },

            rotate: function(cellid, prePos, curPos) {
                var binding_func =
                    (function(err, root) {
                        if (err) {
                            console.log('load proto failed!');
                            throw err;
                        }

                        var curTick = new Date().getTime();
                        if (Math.abs(window.FE.rotateTik - curTick) < 10) {
                            return;
                        }
                        window.FE.rotateTik = curTick;

                        var MsgMouse = root.lookup('medical_imaging.MsgMouse');
                        var msgMouse = MsgMouse.create({
                            pre: { x: prePos.x, y: prePos.y },
                            cur: { x: curPos.x, y: curPos.y },
                            tag: 0
                        });
                        var msgBuffer = MsgMouse.encode(msgMouse).finish();
                        var msgLength = msgBuffer.byteLength;
                        var cmdBuffer = new ArrayBuffer(32 + msgLength);

                        // header
                        var header = new Uint32Array(cmdBuffer, 0, 8);
                        header[0] = 0;
                        header[1] = 0;
                        header[2] = COMMAND_ID_FE_OPERATION;
                        header[3] = cellid;
                        header[4] = OPERATION_ID_ROTATE;
                        header[5] = 0;
                        header[6] = 0;
                        header[7] = msgLength;

                        // data
                        var srcBuffer = new Uint8Array(msgBuffer);
                        var dstBuffer = new Uint8Array(cmdBuffer, 8 * 4, msgLength);
                        for (var index = 0; index < msgLength; index++) {
                            dstBuffer[index] = srcBuffer[index];
                        }
                        console.log('emit paging message.');

                        this.socket.emit('data', {
                            userid: this.userid,
                            username: this.username,
                            content: cmdBuffer
                        });
                    }).bind(this);

                protobuf.load('./data/mi_message.proto', binding_func);
            },

            triggerOnBE: function() {
                var binding_func = (function(err, root) {
                    if (err) {
                        console.log('load proto failed!');
                        throw err;
                    }
                    var MsgInit = root.lookup('medical_imaging.MsgInit');
                    var COMMAND_ID_FE_READY = 120001;
                    var msgInit = MsgInit.create();
                    msgInit.series_uid = 'test_uid';
                    msgInit.pid = 0;

                    // MPR
                    msgInit.cells.push({
                        id: 0,
                        type: 1,
                        direction: 0,
                        width: cellCanvas[0].width,
                        height: cellCanvas[0].height
                    });

                    msgInit.cells.push({
                        id: 1,
                        type: 1,
                        direction: 1,
                        width: cellCanvas[1].width,
                        height: cellCanvas[1].height
                    });

                    msgInit.cells.push({
                        id: 2,
                        type: 1,
                        direction: 2,
                        width: cellCanvas[2].width,
                        height: cellCanvas[2].height
                    });

                    msgInit.cells.push({
                        id: 3,
                        type: 2,
                        direction: 0,
                        width: cellCanvas[3].width,
                        height: cellCanvas[3].height
                    });

                    var msgBuffer = MsgInit.encode(msgInit).finish();
                    var msgLength = msgBuffer.byteLength;
                    var cmdBuffer = new ArrayBuffer(32 + msgLength);

                    // header
                    var header = new Uint32Array(cmdBuffer, 0, 8);
                    header[0] = 0;
                    header[1] = 0;
                    header[2] = COMMAND_ID_FE_OPERATION;
                    header[3] = 0;
                    header[4] = OPERATION_ID_INIT;
                    header[5] = 0;
                    header[6] = 0;
                    header[7] = msgLength;

                    // data
                    var srcBuffer = new Uint8Array(msgBuffer);
                    var dstBuffer =
                        new Uint8Array(cmdBuffer, 8 * 4, msgLength);
                    for (var index = 0; index < msgLength; index++) {
                        dstBuffer[index] = srcBuffer[index];
                    }
                    console.log('emit paging message.');

                    this.socket.emit('data', {
                        userid: this.userid,
                        username: this.username,
                        content: cmdBuffer
                    });
                }).bind(this);

                protobuf.load('./data/mi_message.proto', binding_func);
            },

            loadSeries: function() {
                var header_buffer = new ArrayBuffer(32);
                var header = new Uint32Array(header_buffer);
                header[0] = 0;
                header[1] = 0;
                header[2] = COMMAND_ID_FE_LOAD_SERIES;
                header[3] = 0; // paging
                header[4] = 0;
                header[5] = 0;
                header[6] = 0;
                header[7] = 0;
                this.socket.emit('data', {
                    userid: this.userid,
                    username: this.username,
                    content: header_buffer
                })
            },

            // changeLayout1x1: function() {
            //     document.getElementById("cell1").style.visibility = "hidden";
            //     document.getElementById("cell2").style.visibility = "hidden";
            //     document.getElementById("cell3").style.visibility = "hidden";

            //     var cell0 = document.getElementById("cell0").style.visibility =
            //     "visible";
            //     cell0.width = 1040;
            //     cell0.height = 1024;

            //     document.getElementById("cell0Canvas").width = 1024;
            //     document.getElementById("cell0Canvas").height = 1024;
            // },

            // changeLayout2x2: function() {
            //     document.getElementById("cell1").style.visibility = "visible";
            //     document.getElementById("cell2").style.visibility = "visible";
            //     document.getElementById("cell3").style.visibility = "visible";

            //     var cell0 = document.getElementById("cell0").style.visibility =
            //     "visible";
            //     cell0.width = 518;
            //     cell0.height = 518;

            //     document.getElementById("cell0Canvas").width = 512;
            //     document.getElementById("cell0Canvas").height = 512;
            // }

        },

        window.LOGIC = {

            drawImg: function() {
                for (var i = 0; i < myCanvasImg.data.length; i += 4) {
                    myCanvasImg.data[i] =
                        255; //返回一个对象，其包含指定的 ImageData 对象的图像数据
                    myCanvasImg.data[i + 1] = 255;
                    myCanvasImg.data[i + 2] = 0;
                    myCanvasImg.data[i + 3] = 255;
                }
                myCtx.putImageData(
                    myCanvasImg, 0, 0); //把图像数据（从指定的 ImageData 对象）放回画布上
            },


        },

        window.FE.init('wr')



})()