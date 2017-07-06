//用sockio来构造server（本质上还是HTTPserver）
var app = require("express")();
var http = require("http").Server(app);
var io = require("socket.io")(http);
var util = require("util");
var fs = require("fs");
var child_process = require("child_process");
var body_parser = require("body-parser");

//添加静态目录
app.use(require("express").static(__dirname + "/public"));

var uRLEncoderParser = body_parser.urlencoded({extended:false});

//主页的响应
app.get("/index.html" , function(req, res){
    res.sendFile(__dirname + "/index.html");
    console.log("get index.html");
});

//登录表单的提交
// app.post("/login_in" , uRLEncoderParser , function(req , res){
//     //这里查数据库，匹配用户名和密码
//     if(req.body.username == "wr" && req.body.password == "000000")
//     {
//         res.sendFile(__dirname + "/review.html");
//     }
//     console.log(req.body.username + " has login in.");

//     //建立websocket长连接
// });


//Websocket 长连接用户
var onlineUsers = {};
var onlineLocalSockets = {};
var onlineLocalIPC = {};
var onlineLogicProcessID = {};
var onlineCount = 0;

//FE to BE
var COMMAND_ID_FE_READY = 120001;
var COMMAND_ID_FE_OPERATION = 120002;
var COMMAND_ID_FE_SHUT_DOWN = 121112;
var COMMAND_ID_FE_LOAD_SERIES = 120003;
//BE to FE
var COMMAND_ID_BE_READY = 270001;
var COMMAND_ID_BE_SEND_IMAGE = 270002;


io.on("connection" , function(socket){
    console.log("<><><><><><> connecting <><><><><><>");

    var msgEnd = true;
    var msgLen = 0;
    var msgRest = 0;

    var ipc_sender = 0;
    var ipc_receiver = 0;
    var ipc_msg_id = 0;
    var ipc_msg_info0 = 0;
    var ipc_msg_info1 = 0;
    var ipc_data_type = 0;
    var ipc_big_end = 0;
    var ipc_data_len =0;

    //监听用户加入
    socket.on("login" , function(obj){
        if(obj.username == "")
        {
            console.log("invalid null username!");
            return;
        }

        socket.name = obj.userid;//标示socket

        console.log("<><><><><><> logining <><><><><><>");
        console.log(obj.username + " has login.");
        console.log("id : " + obj.userid);

        if(!onlineUsers.hasOwnProperty(obj.userid)) {
			onlineUsers[obj.userid] = obj.username;
			onlineCount++;
		}
        else
        {
            console.log(obj.username + " is online now.");
        }
        
        
        ///////////////////////////////////////
        //1 建立一个FE IPC实例 用来和即将建立的C++业务逻辑进程通信
        var node_ipc=require('node-ipc');
        const ipc=new node_ipc.IPC;
    
		console.log("UNIX path is : " + obj.username);
        ipc.config.id = obj.username;
        ipc.config.retry= 1500;
        ipc.config.rawBuffer=true;
        ipc.config.encoding='hex';

        ipc.serve(function(){
    
            ipc.server.on('connect' , function(local_socket){
                ///////////////////////////////////////
                //这里在生产环境下要注意，虽然一个逻辑是ipc仅仅构造一个C++逻辑进程，但这样做也是不合适的
                ///////////////////////////////////////
                console.log('IPC connect');
                onlineLocalSockets[obj.userid] = local_socket;
            });

            ///////////////////////////////////////
            //FE socket 收到BE的数据然后转发给Web FE
            //需要解析包以及发包
            ///////////////////////////////////////
            ipc.server.on('data',function(buffer,local_socket){
                
                //ipc.log('got a data : ' + buffer.length);
                //解析下一个报文
                if(msgEnd){
                    //解析IPC data header 32 byte
                    //unsigned int _sender;//sender pid
                    //unsigned int _receiver;//receiver pid
                    //unsigned int _msg_id;//message ID : thus command ID
                    //unsigned int _msg_info0;//message info : thus cell ID
                    //unsigned int _msg_info1;//message info : thus operation ID
                    //unsigned int _data_type;//0 raw_data 1 protocol buffer
                    //unsigned int _big_end;//0 small end 1 big_end 
                    //unsigned int _data_len;//data length

                    ipc_sender = buffer.readUintLE(0,4);
                    ipc_receiver = buffer.readUintLE(4,4);
                    ipc_msg_id = buffer.readUintLE(8,4);
                    ipc_msg_info0 = buffer.readUintLE(12,4);
                    ipc_msg_info1 = buffer.readUintLE(16,4);
                    ipc_data_type = buffer.readUintLE(20,4);
                    ipc_big_end = buffer.readUintLE(24,4);
                    ipc_data_len = buffer.readUintLE(28,4);
                    
                    msgLen = ipc_data_len;
                    msgRest = ipc_data_len;
                    msgEnd = false;

                    //console.log("receive be message : tag " + msgTag);
                    socket.emit("data", buffer)
                    msgRest -= (buffer.length - 32);
                    //一次报文包含了所有信息，不需要分包发送
                    if(msgRest <=0 ){
                        msgEnd = true;
                        msgLen = 0;
                    }
                    
                }
                //持续传递上一个报文
                else
                {
                    //console.log("sending message : " + buffer.length);
                    socket.emit("data", buffer);

                    msgRest -= buffer.length;
                    if(msgRest <=0 ){
                        msgEnd = true;
                        msgLen = 0;
                    }
                }
            });


            ipc.server.on('socket.disconnected', function(local_socket, destroyedSocketID) {
                    ipc.log('client ' + destroyedSocketID + ' has disconnected!');
            });

        });
		ipc.server.start();
        onlineLocalIPC[obj.userid] = ipc;
        ///////////////////////////////////////

        ///////////////////////////////////////
        //2 建立一个C++业务逻辑后台进程

        console.log("path: " + "/tmp/app." +obj.username);
        
        //TODO Test 自己起C++ BE
        // const out = fs.openSync('./out_'+util.inspect(obj.username) + '.log', 'a');
	    // const err = fs.openSync('./err_'+util.inspect(obj.username) + '.log', 'a');
        // var worker = child_process.spawn('./public/cpp/be' ,["/tmp/app."+obj.username] ,{detached: true, stdio: [ 'ignore', out, err ]});
        // onlineLogicProcessID[obj.userid] = worker;

        console.log("<><><><><><> logining success <><><><><><>");

    });

    //监听用户退出
    socket.on("disconnect" , function(obj){

        if(obj.username == ""){
            console.log("invalid null username when disconnect!");
            return;
        }

        var userid = socket.name;
        if(!onlineUsers.hasOwnProperty(userid)){
            console.log(obj.username + " disconnecting failed.");
            return;
        }

        console.log(obj.username + " has disconnecting.");
        console.log("id : " +userid);

        //发最后一个消息给c++ 业务逻辑进程来关闭该进程
        var ipc = onlineLocalIPC[userid];
        var local_socket = onlineLocalSockets[userid];
        //ipc.config.encoding='hex';

        const quitMsg = new Buffer(32);
        quitMsg.writeUIntLE(1,0,4);
        quitMsg.writeUIntLE(0,4,4);
        quitMsg.writeUIntLE(COMMAND_ID_FE_SHUT_DOWN,8,4);
        quitMsg.writeUIntLE(0,12,4);
        quitMsg.writeUIntLE(0,16,4);
        quitMsg.writeUIntLE(0,20,4);
        quitMsg.writeUIntLE(0,24,4);
        quitMsg.writeUIntLE(0,28,4);

        
        ipc.server.emit(local_socket , quitMsg);
        ipc.server.stop();//关闭IPC
		
		//删除user的信息
		delete onlineUsers[userid];
        delete onlineLocalSockets[userid];
        //delete onlineLogicProcessID[userid];
        delete onlineLocalIPC[userid];
		onlineCount--;

        console.log(obj.username + " disconnecting success.");
    });


    //监听用户消息
    socket.on("message" , function(obj){
        console.log(obj.username + " message : " + obj.content);
    });

    socket.on("data" , function(obj){
        userid = obj.userid;
        var ipc = onlineLocalIPC[userid];
        var local_socket = onlineLocalSockets[userid];
        console.log("web send to server : username " + obj.username);
        var buffer = obj.content;
    
        //这里是hardcoding　强制发送paging 的 operation,还不是非常清楚如何讲web传递过来的数据转换成buffer传递出去
        var command_id = buffer[2];
        if(command_id == COMMAND_ID_FE_OPERATION){
            const msgBE = new Buffer(32);
            msgBE.writeUIntLE(COMMAND_ID_FE_OPERATION,8,4);
            ipc.server.emit(local_socket , msgBE);
        }
        else if(command_id == COMMAND_ID_FE_LOAD_SERIES){
            const msgBE = new Buffer(32);
            msgBE.writeUIntLE(COMMAND_ID_FE_LOAD_SERIES,8,4);
            ipc.server.emit(local_socket , msgBE);
        }

        //ipc.server.emit(local_socket , buffer);
    });
    
    
})



var server = http.listen(8080,function(){
    var address = server.address();
    console.log("address is " , util.inspect(address));
});
