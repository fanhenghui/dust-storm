var fs = require('fs');
var childProcess = require('child_process');

var COMMAND_ID_FE_SHUT_DOWN = 120000;
var COMMAND_ID_FE_READY = 120001;
var COMMAND_ID_FE_HEARTBEAT = 119999;

var onlineUsers = {};
var onlineLocalSockets = {};
var onlineLocalIPC = {};
var onlineLogicProcessID = {};
var onlineBEHBTic = {};
var onlineFEHBTic = {};
var onlineCount = 0;

module.exports = {
  onIOSocketConnect: function(socket) {
    console.log('<><><><><><> connecting <><><><><><>');
    //监听用户加入
    socket.on('login', function(obj) {
      if (obj.username == '') {
        console.log('invalid null username!');
        return;
      }

      socket.name = obj.userid;  // uid to locate web socket
      console.log('<><><><><><> logining <><><><><><>');
      console.log(obj.username + ' has login.');
      console.log('id : ' + obj.userid);

      if (!onlineUsers.hasOwnProperty(obj.userid)) {
        onlineUsers[obj.userid] = obj.username;
        onlineCount++;
      } else {
        console.log(obj.username + ' is online now.');
      }

      // create a server IPC between server and BE
      var nodeIpc = require('node-ipc');
      const ipc = new nodeIpc.IPC;

      console.log('UNIX path is : ' + obj.username);
      ipc.config.id = obj.username;
      ipc.config.retry = 1500;
      ipc.config.rawBuffer = true;
      ipc.config.encoding = 'hex';
      ipc.serve(function() { 
        ipc.server.on('connect', function(localSocket) {
          // when IPC is constructed between server and BE
          // server send a ready MSG to BE to make sure BE is ready
          console.log('IPC connect. Send FE ready to BE.');

          const msgFEReady = new Buffer(32);
          msgFEReady.writeUIntLE(1, 0, 4);
          msgFEReady.writeUIntLE(0, 4, 4);
          msgFEReady.writeUIntLE(COMMAND_ID_FE_READY, 8, 4);
          msgFEReady.writeUIntLE(0, 12, 4);
          msgFEReady.writeUIntLE(0, 16, 4);
          msgFEReady.writeUIntLE(0, 20, 4);
          msgFEReady.writeUIntLE(0, 24, 4);
          msgFEReady.writeUIntLE(0, 28, 4);

          if (ipc != undefined) {
            ipc.server.emit(localSocket, msgFEReady);
          }
          onlineLocalSockets[obj.userid] = localSocket;

          // send heart bet package
          var heartBeatBE = (function() {
            if (onlineLocalIPC.hasOwnProperty(obj.userid) && onlineLocalSockets.hasOwnProperty(obj.userid)) {
              const msgHeartBeat = new Buffer(32);
              msgHeartBeat.writeUIntLE(1, 0, 4);
              msgHeartBeat.writeUIntLE(0, 4, 4);
              msgHeartBeat.writeUIntLE(COMMAND_ID_FE_HEARTBEAT, 8, 4);
              msgHeartBeat.writeUIntLE(0, 12, 4);
              msgHeartBeat.writeUIntLE(0, 16, 4);
              msgHeartBeat.writeUIntLE(0, 20, 4);
              msgHeartBeat.writeUIntLE(0, 24, 4);
              msgHeartBeat.writeUIntLE(0, 28, 4);
              onlineLocalIPC[obj.userid].server.emit(onlineLocalSockets[obj.userid], msgHeartBeat);
              console.log("server heart beat for user: " + obj.username);
            }
          });
          const heartBeatTime = 2 * 1000;
          onlineBEHBTic[obj.userid] = setInterval(heartBeatBE, heartBeatTime);
        });

        // sever get logic process's tcp package and transfer to FE directly(not parse)
        ipc.server.on('data', function(buffer, localSocket) {
          ipc.log('got a data : ' + buffer.length);
          socket.emit('data', buffer);
        });

        ipc.server.on('socket.disconnected', function(localSocket, destroyedSocketID) {
          ipc.log('client ' + destroyedSocketID + ' has disconnected!');
        });
      });

      ipc.server.start();
      onlineLocalIPC[obj.userid] = ipc;

      // Create BE process
      console.log('path: ' + '/tmp/app.' + obj.username);
      // reader BE process path form config file
      var review_server_path;
      fs.readFile(__dirname + '/be_path', function(err, data) {
        if (err) { 
          throw err;
        }
        review_server_path = data.toString();
        console.log('app path : ' + review_server_path);

        /// run process
        var worker = childProcess.spawn( review_server_path, ['/tmp/app.' + obj.username], {detached: true});
        onlineLogicProcessID[obj.userid] = worker;

        //// std ouput to server device
        worker.stdout.on('data', (data) => {
          // console.log(`stdout: ${data}`);
        });
        worker.stderr.on('data', (data) => {
          // console.log(`stderr: ${data}`);
        });

        worker.on('close', (code) => {
          console.log(`child process exited with code ${code}`);
        });
        console.log('<><><><><><> login in success <><><><><><>');
      });
    });

    // listen user disconnect(leave review page)
    socket.on('disconnect', function(obj) {
      if (obj.username == '') {
        console.log('invalid null username when disconnect!');
        return;
      }
      var userid = socket.name;
      if (!onlineUsers.hasOwnProperty(userid)) {
        console.log(obj.username + ' disconnecting failed.');
        return;
      }

      console.log(obj.username + ' has disconnecting.');
      console.log('id : ' + userid);

      // send last msg tp BE to notify BE shut down
      var ipc = onlineLocalIPC[userid];
      var localSocket = onlineLocalSockets[userid];

      const quitMsg = new Buffer(32);
      quitMsg.writeUIntLE(1, 0, 4);
      quitMsg.writeUIntLE(0, 4, 4);
      quitMsg.writeUIntLE(COMMAND_ID_FE_SHUT_DOWN, 8, 4);
      quitMsg.writeUIntLE(0, 12, 4);
      quitMsg.writeUIntLE(0, 16, 4);
      quitMsg.writeUIntLE(0, 20, 4);
      quitMsg.writeUIntLE(0, 24, 4);
      quitMsg.writeUIntLE(0, 28, 4);

      if (ipc != undefined) {
        ipc.server.emit(localSocket, quitMsg);
        ipc.server.stop();
      }

      // purge user's infos
      if (onlineUsers.hasOwnProperty(userid)) {
        delete onlineUsers[userid];
      }
      if (onlineLocalSockets.hasOwnProperty(userid)) {
        delete onlineLocalSockets[userid];
      }
      if(onlineLocalIPC.hasOwnProperty(userid)) {
        delete onlineLocalIPC[userid];
      }     
      if(onlineBEHBTic.hasOwnProperty(userid)) {
        clearInterval(onlineBEHBTic[userid]);
        delete onlineBEHBTic[userid];
      }
      onlineCount--;
      console.log(obj.username + ' disconnecting success.');
    });

    // listen string message for test
    socket.on('message', function(obj) {
      console.log(obj.username + ' message : ' + obj.content);
    });

    socket.on('data', function(obj) {
      // TODO FE 的　TCP 粘包问题是在BE端解?
      userid = obj.userid;
      console.log('socket on data , userid : ' + userid);
      var ipc = onlineLocalIPC[userid];
      if (ipc === undefined || ipc == null) {
        console.log('socket on data , ERROR IPC');
        return;
      }

      var localSocket = onlineLocalSockets[userid];
      if (localSocket === undefined || localSocket == null) {
        console.log('socket on data , ERROR SOCKET');
        return;
      }

      console.log('web send to server : username ' + obj.username);
      var buffer = obj.content;
      var commandID = buffer.readUIntLE(8, 4);

      console.log('buffer length : ' + buffer.byteLength);
      console.log('command id :' + commandID);

      ipc.server.emit(localSocket, buffer);
    });
  }
}