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

io.on("connection" , function(socket){
    console.log("<><><><><><> connecting <><><><><><>");

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
    
        ipc.serve(function(){
    
            ipc.config.rawBuffer=true;

            ipc.server.on('connect' , function(local_socket){
                console.log('IPC connect');
                onlineLocalSockets[obj.userid] = local_socket;
            });

            ipc.server.on('data',function(buffer,local_socket){
                
                ipc.log('got a data : ' + buffer.length);
                buffer.slice()
                ipc.log(buffer.toString('ascii'));
            });

            ipc.server.on('socket.disconnected', function(local_socket, destroyedSocketID) {
                    ipc.log('client ' + destroyedSocketID + ' has disconnected!');
            });

        });
		ipc.server.start();
        onlineLocalIPC[obj.userid] = ipc;

        //ipc.server.stop();
        ///////////////////////////////////////

        ///////////////////////////////////////
        //2 建立一个C++业务逻辑后台进程
        const out = fs.openSync('./out_'+util.inspect(obj.username) + '.log', 'a');
	    const err = fs.openSync('./err_'+util.inspect(obj.username) + '.log', 'a');
        var worker = child_process.spawn('./public/cpp/be' ,["/tmp/app."+obj.username] ,{detached: true, stdio: [ 'ignore', out, err ]});
        onlineLogicProcessID[obj.userid] = worker;

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

        //发最后一个消息给c++ 业务逻辑进程
        var ipc = onlineLocalIPC[userid];
        var local_socket = onlineLocalSockets[userid];
        //ipc.config.encoding='hex';

        const quitMsg = new Buffer(16);
        quitMsg.writeIntLE(-1,0,4);
        quitMsg.writeIntLE(0,4,4);
        quitMsg.writeIntLE(0,8,8);
        
        ipc.server.emit(local_socket , quitMsg);
        ipc.server.stop();//关闭IPC
		
		//删除user的信息
		delete onlineUsers[userid];
        delete onlineLocalSockets[userid];
        delete onlineLogicProcessID[userid];
		onlineCount--;

        console.log(obj.username + " disconnecting success.");
    });


    //监听用户消息
    socket.on("message" , function(obj){
        console.log(obj.username + " message : " + obj.content);
    });
    
})



var server = http.listen(8080,function(){
    var address = server.address();
    console.log("address is " , util.inspect(address));
});
