(function() {

    myCanvas = document.getElementById("myCanvas");
    myCtx = myCanvas.getContext("2d");
    //myCanvasImg = myCtx.createImageData(myCanvas.width, myCanvas.height);
    myCanvasImgJpeg = new Image();

    msgEnd = true;
    msgRest = 0;
    msgLen = 0;

    ipc_sender = 0;
    ipc_receiver = 0;
    ipc_msg_id = 0;
    ipc_msg_info0 = 0;
    ipc_msg_info1 = 0;
    ipc_data_type = 0;
    ipc_big_end = 0;
    ipc_data_len = 0;


    //FE to BE
    COMMAND_ID_FE_READY = 120001;
    COMMAND_ID_FE_OPERATION = 120002;
    COMMAND_ID_FE_SHUT_DOWN = 121112;
    COMMAND_ID_FE_LOAD_SERIES = 120003;
    COMMAND_ID_FE_MPR_PAGE = 120004;

    //BE to FE
    COMMAND_ID_BE_READY = 270001;
    COMMAND_ID_BE_SEND_IMAGE = 270002;




    window.FE = {
        username: null,
        userid: null,
        socket: null,

        genUID: function(username) {
            return username + new Date().getTime() + "" + Math.floor(Math.random() * 173 + 511);
        },

        init: function(username) {
            //客户端根据时间和随机数生成uid,以用户名称可以重复，后续改为数据库的UID名称
            this.userid = this.genUID(username);
            this.username = username;

            //链接websocket服务器
            this.socket = io.connect("http://172.23.237.249:8080");

            //通知服务器有用户登录
            this.socket.emit("login", { userid: this.userid, username: this.username });

            //发送一段message
            this.socket.emit("message", { userid: this.userid, username: this.username, content: "first message" });



            // this.socket.on("tick" , function(obj))
            // {

            // }


            this.socket.on("data", function(arraybuffer) {
                curImgLen = arraybuffer.byteLength;
                bufferOffset = 0;
                if (msgEnd) {
                    //data command handler
                    //解析IPC data header 32 byte
                    //unsigned int _sender;//sender pid
                    //unsigned int _receiver;//receiver pid
                    //unsigned int _msg_id;//message ID : thus command ID
                    //unsigned int _msg_info0;//message info : thus cell ID
                    //unsigned int _msg_info1;//message info : thus operation ID
                    //unsigned int _data_type;//0 raw_data 1 protocol buffer
                    //unsigned int _big_end;//0 small end 1 big_end 
                    //unsigned int _data_len;//data length
                    var header = new Uint32Array(arraybuffer, 0, 8);

                    ipc_sender = header[0];
                    ipc_receiver = header[1];
                    ipc_msg_id = header[2];
                    ipc_msg_info0 = header[3];
                    ipc_msg_info1 = header[4];
                    ipc_data_type = header[5];
                    ipc_big_end = header[6];
                    ipc_data_len = header[7];

                    msgLen = ipc_data_len;
                    msgRest = msgLen;
                    msgEnd = false;

                    curImgLen = arraybuffer.byteLength - 32;
                    bufferOffset = 32;
                }


                if (curImgLen > 0) {
                    //Handle data
                    if (ipc_msg_id == COMMAND_ID_BE_READY) {
                        console.log("Ready");
                        //this.socket.emit
                    } else if (ipc_msg_id == COMMAND_ID_BE_SEND_IMAGE) {
                        //Draw jpeg buffer
                        var imgBuffer = new Uint8Array(arraybuffer, bufferOffset, curImgLen);
                        var b64encoded = btoa(String.fromCharCode.apply(null, imgBuffer));
                        var datajpg = "data:image/jpg;base64," + b64encoded;

                        //var img = new Image();
                        myCanvasImgJpeg.src = datajpg;
                        myCanvasImgJpeg.onload = function() {
                            console.log("Image Onload");
                            myCtx.drawImage(myCanvasImgJpeg, 0, 0, myCanvas.width, myCanvas.height);
                        };

                        ////Draw raw image 
                        //var imgBuffer = new Uint8Array(arraybuffer , bufferOffset, curImgLen);
                        // for(var i = 0 ; i< curImgLen ; ++i){
                        // myCanvasImg.data[msgLen - msgRest + i] = imgBuffer[i];
                        // }
                    }

                }

                msgRest -= curImgLen;
                if (msgRest <= 0) {
                    msgRest = 0;
                    msgLen = 0;
                    msgEnd = true;
                    myCtx.putImageData(myCanvasImg, 0, 0);
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
            var header_buffer = new ArrayBuffer(32);
            var header = new Uint32Array(header_buffer);
            header[0] = 0;
            header[1] = 0;
            header[2] = COMMAND_ID_FE_MPR_PAGE;
            header[3] = 11; //paging
            header[4] = 0;
            header[5] = 0;
            header[6] = 0;
            header[7] = 0;
            this.socket.emit("data", { userid: this.userid, username: this.username, content: header })
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
            this.socket.emit("data", { userid: this.userid, username: this.username, content: header })
        }
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




})()