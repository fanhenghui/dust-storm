var app = require('express')();
var http = require('http').Server(app);
var io = require('socket.io')(http);
var fs = require("fs");
var util = require("util");
var child_process = require("child_process");

app.use(require('express').static(__dirname + "/public"));

app.get('/', function(req, res){
	//res.send('<h1>Welcome Realtime Server</h1>');
	res.sendFile(__dirname + "/" +"index.html");
});


//在线用户
var onlineUsers = {};
//当前在线人数
var onlineCount = 0;

io.on('connection', function(socket){
	console.log('a user connected');
	
	//监听新用户加入
	socket.on('login', function(obj){
		//将新加入用户的唯一标识当作socket的名称，后面退出的时候会用到
		socket.name = obj.userid;

		///////////////////////////////////////
	    //Test 1 process 启动一个后台进程
	    const out = fs.openSync('./out_'+util.inspect(obj.username) + '.log', 'a');
	    const err = fs.openSync('./err_'+util.inspect(obj.username) + '.log', 'a');
	    var worker = child_process.spawn('./talker' ,[obj.username] ,{detached: true, stdio: [ 'ignore', out, err ]});
		///////////////////////////////////////

		///////////////////////////////////////
	    //Test 2 process 启动一个后台进程，并且能够和websocket服务器通信
		var node_ipc=require('node-ipc');

        const ipc=new node_ipc.IPC;
    
		console.log("path is : " + obj.username);
        ipc.config.id   = obj.username;//'world' + i;
        ipc.config.retry= 1500;
    
        ipc.serve(function(){
    
            ipc.config.rawBuffer=true;
            ipc.server.on('data',function(buffer,local_socket){
                
                ipc.log('got a data : ' + buffer.length);
                ipc.log(buffer.toString('ascii'));
            });
    
            ipc.server.on('message',function(data,local_socket){
                
                ipc.log('got a message : ', data);
                ipc.server.emit(local_socket,'message',  //this can be anything you want so long as your client knows. 
                        data+' world!');
            });
    
            ipc.server.on('socket.disconnected', function(local_socket, destroyedSocketID) {
                    ipc.log('client ' + destroyedSocketID + ' has disconnected!');
            });
        });
		ipc.server.start();
		///////////////////////////////////////


		//检查在线列表，如果不在里面就加入
		if(!onlineUsers.hasOwnProperty(obj.userid)) {
			onlineUsers[obj.userid] = obj.username;
			//在线人数+1
			onlineCount++;
		}
		
		//向所有客户端广播用户加入
		io.emit('login', {onlineUsers:onlineUsers, onlineCount:onlineCount, user:obj});
		console.log(obj.username+'加入了聊天室');
	});
	
	//监听用户退出
	socket.on('disconnect', function(){
		//将退出的用户从在线列表中删除
		if(onlineUsers.hasOwnProperty(socket.name)) {
			//退出用户的信息
			var obj = {userid:socket.name, username:onlineUsers[socket.name]};
			
			//删除
			delete onlineUsers[socket.name];
			//在线人数-1
			onlineCount--;
			
			//向所有客户端广播用户退出
			io.emit('logout', {onlineUsers:onlineUsers, onlineCount:onlineCount, user:obj});
			console.log(obj.username+'退出了聊天室');
		}
	});
	
	//监听用户发布聊天内容
	socket.on('message', function(obj){
		//向所有客户端广播发布的消息
		io.emit('message', obj);
		console.log(obj.username+'说：'+obj.content);
	});
  
});

http.listen(8000, function(){
	console.log('listening on *:8000');
});
