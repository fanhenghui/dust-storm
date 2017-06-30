(function(){

myCanvas = document.getElementById("myCanvas");
myCtx = myCanvas.getContext("2d");
myCanvasImg = myCtx.createImageData(myCanvas.width , myCanvas.height);

msgEnd = true;
msgRest = 0;
msgLen = 0;
msgTag = 0;

window.FE = {
    username:null,
    userid:null,
    socket:null,

    genUID:function(username)
    {
        return username + new Date().getTime() + ""+Math.floor(Math.random()*173+511);
    },

    init:function(username)
    {
        //客户端根据时间和随机数生成uid,以用户名称可以重复，后续改为数据库的UID名称
        this.userid = this.genUID(username);
        this.username = username;

        //链接websocket服务器
        this.socket = io.connect("http://172.23.236.115:8080");

        //通知服务器有用户登录
        this.socket.emit("login" , {userid:this.userid , username:this.username});

        //发送一段message
        this.socket.emit("message",{userid:this.userid , username:this.username , content:"first message"});

        // this.socket.on("tick" , function(obj))
        // {

        // }

        this.socket.on("talk",function(arraybuffer){
            console.log(arraybuffer.byteLength);
            // var tag = new Int32Array(arraybuffer,0,1);
            // console.log("tag : " + tag[0]);
            // var len = new Int32Array(arraybuffer,4,1);
            // console.log("length : " + len[0]);
            
            // if(tag == 0){   
            //     var msg = new Int8Array(arraybuffer , 16,len[0]);
            //     var s = new String;
            //     for(var i = 0 ; i< len ; ++i)
            //     {
            //         s += msg[i];
            //     }
            //     console.log(s);
            // }
            
        });


        this.socket.on("image" , function(arraybuffer){

            curImgLen = arraybuffer.byteLength;
            bufferOffset = 0;
            if(msgEnd){
                var tag = new Int32Array(arraybuffer,0,1);
                //console.log("tag : " + tag[0]);
                var len = new Int32Array(arraybuffer,4,1);
                //console.log("length : " + len[0]);

                msgTag = tag[0];
                msgLen = len[0];lastPixel = 0;
                msgRest = msgLen;
                msgEnd = false;

                curImgLen = arraybuffer.byteLength - 16;
                bufferOffset = 16;
            }

            if(curImgLen > 0){
                var imgBuffer = new Uint8Array(arraybuffer , bufferOffset, curImgLen);
                for(var i = 0 ; i< curImgLen ; ++i){
                    myCanvasImg.data[msgLen - msgRest + i] = imgBuffer[i];
                }
            }

            msgRest -= curImgLen;
            if(msgRest <= 0){
                msgRest = 0;
                msgTag = -1;
                msgLen = 0;
                msgEnd = true;
                myCtx.putImageData(myCanvasImg,0,0);
            }

            // var tag = new Int32Array(arraybuffer,0,1);
            // console.log("tag : " + tag[0]);
            // var len = new Int32Array(arraybuffer,4,1);
            // console.log("length : " + len[0]);
            // if(tag == 1){
            //     if(len[0] == myCanvasImg.data.length){
            //         console.log("ready to draw ... ");
            //         var imgBuffer = new Uint8Array(arraybuffer , 16, len[0]);
            //         for(var i = 0;i<myCanvasImg.data.length;++i){
            //             myCanvasImg.data[i] = imgBuffer[i];
            //         }
            //         myCtx.putImageData(myCanvasImg,0,0);
            //         console.log("draw end ... ");
            //     }
            // }
        });

    },

    userLogOut:function()
    {
        //this.socket.emit("logout",{userid:this.userid , username:this.username , content:"last message"});
        //this.socket.emit("disconnect" , {userid:this.userid , username:this.username});
        location.reload();
    },


    userLogIn:function()
    {
        var username = document.getElementById("username").value;
        
        this.init(username);
    },
}

window.LOGIC = {
    
    drawImg:function()
    {
        for (var i=0;i<myCanvasImg.data.length;i+=4){
            myCanvasImg.data[i]=255;//返回一个对象，其包含指定的 ImageData 对象的图像数据
            myCanvasImg.data[i+1]=255;
            myCanvasImg.data[i+2]=0;
            myCanvasImg.data[i+3]=255;
        }
        myCtx.putImageData(myCanvasImg,0,0);//把图像数据（从指定的 ImageData 对象）放回画布上
    },

}




})()