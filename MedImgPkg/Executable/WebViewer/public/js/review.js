(function() {
    msgEnd = true;
    msgRest = 0;
    msgLen = 0;

    //FE to BE
    COMMAND_ID_FE_SHUT_DOWN = 120000;
    COMMAND_ID_FE_READY = 120001;
    COMMAND_ID_FE_OPERATION = 120002;
    COMMAND_ID_FE_MPR_PLAY = 120003;
    COMMAND_ID_FE_VR_PLAY = 120004;

    //BE to FE
    COMMAND_ID_BE_SEND_IMAGE = 270001;
    COMMAND_ID_BE_READY = 270000;

    //FE to BE Operation ID
    OPERATION_ID_INIT = 310000;
    OPERATION_ID_MPR_PAGING = 310001;
    OPERATION_ID_PAN = 310002;
    OPERATION_ID_ZOOM = 310003;
    OPERATION_ID_ROTATE = 310004;

    //init
    cellCanvas = [
        document.getElementById("canvas0"),
        document.getElementById("canvas1"),
        document.getElementById("canvas2"),
        document.getElementById("canvas3")
    ];

    cellJpeg = "";
    cellImage = new Image();
    myDataJpeg = null;

    //init canvas size
    console.log("w:" + window.innerWidth);
    console.log("h:" + window.innerHeight);

    cellContainerWidth = document.getElementById("cell-container").offsetWidth;
    cellContainerHeight = document.getElementById("cell-container").offsetHeight;
    navigatorHeight = document.getElementById("navigator-div").offsetHeight;

    console.log("cells w:" + cellContainerWidth);
    console.log("cells h:" + cellContainerHeight);

    for (var index = 0; index < cellCanvas.length; index++) {
        cellCanvas[index].width = (cellContainerWidth - 20) / 2;
        cellCanvas[index].height = (window.innerHeight - navigatorHeight - 40) / 2;

    }

    function renderToCanvas(curImgLen, bufferOffset, arrayBuffer, cellID) {
        console.log("in render");
        //Construct jpeg data
        if (bufferOffset == 32) {
            myDataJpeg = "";
        } else {
            //console.log("second input");
        }
        var imgBuffer = new Uint8Array(arrayBuffer, bufferOffset, curImgLen);
        cellJpeg += String.fromCharCode.apply(null, imgBuffer);

        if (msgRest - curImgLen <= 0) {
            myDataJpeg.src = "data:image/jpg;base64," + btoa(cellJpeg);
            myDataJpeg.onload = function() {
                console.log("Image Onload");
                cellCanvas[cellID].getContext("2d").drawImage(myDataJpeg, 0, 0, cellCanvas[cellID].width, cellCanvas[cellID].height);
            };
        }
    }

    window.FE = {
        username: null,
        userid: null,
        socket: null,
        lastMsgID: 0,
        lastMsgCellID: 0,


        resize: function() {
            console.log("resize");

            cellContainerWidth = document.getElementById("cell-container").offsetWidth;
            cellContainerHeight = document.getElementById("cell-container").offsetHeight;
            navigatorHeight = document.getElementById("navigator-div").offsetHeight;

            console.log("cells w:" + cellContainerWidth);
            console.log("cells h:" + cellContainerHeight);

            for (var index = 0; index < cellCanvas.length; index++) {
                cellCanvas[index].width = (cellContainerWidth - 20) / 2;
                cellCanvas[index].height = (window.innerHeight - navigatorHeight - 40) / 2;

            }
        },

        genUID: function(username) {
            return username + new Date().getTime() + "" + Math.floor(Math.random() * 173 + 511);
        },

        init: function(username) {
            //客户端根据时间和随机数生成uid,以用户名称可以重复，后续改为数据库的UID名称
            this.userid = this.genUID(username);
            this.username = username;

            //链接websocket服务器
            this.socket = io.connect("http://172.23.237.208:8000");

            //通知服务器有用户登录 TODO 这段逻辑应该在登录的时候做
            this.socket.emit("login", { userid: this.userid, username: this.username });

            //发送一段message
            this.socket.emit("message", { userid: this.userid, username: this.username, content: "first message" });


            this.socket.on("data", function(arraybuffer) {
                console.log("receive data.")
                curImgLen = arraybuffer.byteLength;
                bufferOffset = 0;
                if (msgEnd) {
                    var header = new Uint32Array(arraybuffer, 0, 8);
                    var ipcSender = header[0];
                    var ipcReceiver = header[1];
                    window.FE.lastMsgID = header[2];
                    window.FE.lastMsgCellID = header[3];
                    var ipcMsgInfo1 = header[4];
                    var ipcDataType = header[5];
                    var ipcBigEnd = header[6];
                    var ipcDataLen = header[7];

                    msgLen = ipcDataLen;
                    msgRest = msgLen;
                    msgEnd = false;

                    curImgLen = arraybuffer.byteLength - 32;
                    bufferOffset = 32;
                }

                if (curImgLen >= 0) {
                    switch (window.FE.lastMsgID) {
                        case COMMAND_ID_BE_READY:
                            console.log("Ready");
                            window.FE.triggerOnBE();
                            break;

                        case COMMAND_ID_BE_SEND_IMAGE:
                            console.log("get BE sending img buffer.")
                            renderToCanvas(curImgLen, bufferOffset, arraybuffer, window.FE.lastMsgCellID);

                        default:
                            break;
                    }
                }

                msgRest -= curImgLen;
                if (msgRest <= 0) {
                    msgRest = 0;
                    msgLen = 0;
                    msgEnd = true;
                }
            });

        },

        userLogOut: function() {
            //this.socket.emit("logout",{userid:this.userid , username:this.username , content:"last message"});
            //this.socket.emit("disconnect" , {userid:this.userid , username:this.username});
            location.reload();
        },

        userLogIn: function() {
            var username = document.getElementById("username").value;

            this.init(username);
        },

        paging: function() {
            var binding_func = (function(err, root) {
                if (err) {
                    console.log("load proto failed!");
                    throw err;
                }
                var MsgPaging = root.lookup("medical_imaging.MsgPaging");
                var msgPaging = MsgPaging.create({ page: 1 });
                var msgBuffer = MsgPaging.encode(msgPaging).finish();
                var msgLength = msgBuffer.byteLength;
                var cmdBuffer = new ArrayBuffer(32 + msgLength);

                //header
                var header = new Uint32Array(cmdBuffer, 0, 8);
                header[0] = 0;
                header[1] = 0;
                header[2] = COMMAND_ID_FE_MPR_PLAY;
                header[3] = 0;
                header[4] = 0;
                header[5] = 0;
                header[6] = 0;
                header[7] = msgLength;

                //data
                var srcBuffer = new Uint8Array(msgBuffer);
                var dstBuffer = new Uint8Array(cmdBuffer, 8 * 4, msgLength)
                for (var index = 0; index < msgLength; index++) {
                    dstBuffer[index] = srcBuffer[index];
                }
                console.log("emit paging message.");

                this.socket.emit("data", { userid: this.userid, username: this.username, content: cmdBuffer });
            }).bind(this);

            protobuf.load("./data/mi_message.proto", binding_func);
        },

        triggerOnBE: function() {
            var binding_func = (function(err, root) {
                if (err) {
                    console.log("load proto failed!");
                    throw err;
                }
                var MsgInit = root.lookup("medical_imaging.MsgInit");
                var msgInit = MsgInit.create();
                msgInit.series_uid = "test_uid";
                msgInit.pid = 0;

                //MPR
                msgInit.cells.push({
                    id: 0,
                    type: 1,
                    direction: 0,
                    width: cellCanvas[0].width,
                    height: cellCanvas[0].height
                });

                // msgInit.cells.push({
                //     id: 1,
                //     type: 1,
                //     direction: 0,
                //     width: cellCanvas[1].width,
                //     height: cellCanvas[1].height
                // });

                // msgInit.cells.push({
                //     id: 2,
                //     type: 1,
                //     direction: 0,
                //     width: cellCanvas[2].width,
                //     height: cellCanvas[2].height
                // });

                // msgInit.cells.push({
                //     id: 3,
                //     type: 1,
                //     direction: 0,
                //     width: cellCanvas[3].width,
                //     height: cellCanvas[3].height
                // });

                var msgBuffer = MsgInit.encode(msgInit).finish();
                var msgLength = msgBuffer.byteLength;
                var cmdBuffer = new ArrayBuffer(32 + msgLength);

                //header
                var header = new Uint32Array(cmdBuffer, 0, 8);
                header[0] = 0;
                header[1] = 0;
                header[2] = COMMAND_ID_FE_OPERATION;
                header[3] = 0;
                header[4] = OPERATION_ID_INIT;
                header[5] = 0;
                header[6] = 0;
                header[7] = msgLength;

                //data
                var srcBuffer = new Uint8Array(msgBuffer);
                var dstBuffer = new Uint8Array(cmdBuffer, 8 * 4, msgLength)
                for (var index = 0; index < msgLength; index++) {
                    dstBuffer[index] = srcBuffer[index];
                }
                console.log("emit paging message.");

                this.socket.emit("data", { userid: this.userid, username: this.username, content: cmdBuffer });
            }).bind(this);

            protobuf.load("./data/mi_message.proto", binding_func);
        },

        loadSeries: function() {
            var header_buffer = new ArrayBuffer(32);
            var header = new Uint32Array(header_buffer);
            header[0] = 0;
            header[1] = 0;
            header[2] = COMMAND_ID_FE_LOAD_SERIES;
            header[3] = 0; //paging
            header[4] = 0;
            header[5] = 0;
            header[6] = 0;
            header[7] = 0;
            this.socket.emit("data", { userid: this.userid, username: this.username, content: header_buffer })
        },

        // changeLayout1x1: function() {
        //     document.getElementById("cell1").style.visibility = "hidden";
        //     document.getElementById("cell2").style.visibility = "hidden";
        //     document.getElementById("cell3").style.visibility = "hidden";

        //     var cell0 = document.getElementById("cell0").style.visibility = "visible";
        //     cell0.width = 1040;
        //     cell0.height = 1024;

        //     document.getElementById("cell0Canvas").width = 1024;
        //     document.getElementById("cell0Canvas").height = 1024;
        // },

        // changeLayout2x2: function() {
        //     document.getElementById("cell1").style.visibility = "visible";
        //     document.getElementById("cell2").style.visibility = "visible";
        //     document.getElementById("cell3").style.visibility = "visible";

        //     var cell0 = document.getElementById("cell0").style.visibility = "visible";
        //     cell0.width = 518;
        //     cell0.height = 518;

        //     document.getElementById("cell0Canvas").width = 512;
        //     document.getElementById("cell0Canvas").height = 512;
        // }

    }

    window.LOGIC = {

        drawImg: function() {
            for (var i = 0; i < myCanvasImg.data.length; i += 4) {
                myCanvasImg.data[i] = 255; //返回一个对象，其包含指定的 ImageData 对象的图像数据
                myCanvasImg.data[i + 1] = 255;
                myCanvasImg.data[i + 2] = 0;
                myCanvasImg.data[i + 3] = 255;
            }
            myCtx.putImageData(myCanvasImg, 0, 0); //把图像数据（从指定的 ImageData 对象）放回画布上
        },


    }

    window.FE.init("wr")



})()