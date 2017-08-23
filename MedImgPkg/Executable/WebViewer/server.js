var app = require('express')();
var path = require('path');
var http = require('http').Server(app);
var io = require('socket.io')(http);
var util = require('util');
var fs = require('fs');
var mongoose = require('mongoose');
var session = require('express-session');
var logger = require('morgan');
var cookieParser = require('cookie-parser');
var multer = require('multer');

var childProcess = require('child_process');
var bodyParser = require('body-parser');

var routes = require('./public/routes/index');

global.dbHandel = require('./public/database/dbHandel');
global.db = mongoose.connect('mongodb://localhost:27017/nodedb');
// use session for login
app.use(session(
    {secret: 'secret', cookie: {maxAge: 1000 * 60 * 30}}));  // 30 min timeout

// simply copy here, figure out later
app.use(logger('dev'));
// 解析 ajax 提交的 request 的 body
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended: true}));
app.use(multer());
app.use(cookieParser());

//添加静态目录
app.use(require('express').static(__dirname + '/public'));

// how to render template files
app.set('views', path.join(__dirname, '/public/views'));
app.engine('html', require('ejs').renderFile);
app.set('view engine', 'html');


// //主页的响应
// app.get('/index.html', function(req, res) {
//   res.sendFile(__dirname + '/index.html');
//   console.log('get index.html');
// });

// app.get('/review.html', function(req, res) {
//   res.sendFile(__dirname + '/review.html');
//   console.log('get review.html');
// });

app.use('/', routes);
//登录表单的提交
// var uRLEncoderParser = bodyParser.urlencoded({extended: false});
// app.post("/login_in" , uRLEncoderParser , function(req , res){
//     //这里查数据库，匹配用户名和密码
//     if(req.body.username == "wr" && req.body.password == "000000")
//     {
//         res.sendFile(__dirname + "/review.html");
//     }
//     console.log(req.body.username + " has login in.");

//     //建立websocket长连接
// });


// Websocket 长连接用户
var onlineUsers = {};
var onlineLocalSockets = {};
var onlineLocalIPC = {};
var onlineLogicProcessID = {};
var onlineCount = 0;

////////////////////////////////////////////////////////////
// server as transmit level . just need to know a little command ID
// FE to BE
var COMMAND_ID_FE_SHUT_DOWN = 120000;
var COMMAND_ID_FE_READY = 120001;
////////////////////////////////////////////////////////////

io.on(
    'connection',
    function(socket) {
      console.log('<><><><><><> connecting <><><><><><>');

      var thisMsgEnd = 0;  // 0 msg header 1 data for last msg 2 msg header for
                           // last msg header
      var msgLen = 0;
      var lastMsgRest = 0;

      var lastMsgDataType = 0;
      var lastMsgBigEnd = 0;
      var lastMsgDataLen = 0;

      var lastMsgCmdID = 0;
      var lastHeader = new Buffer(32);
      var lastHeaderByteLength = 0;

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
          io
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

          ipc.server.on('connect', function(local_socket) {
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
              // close IPC
              ipc.server.emit(local_socket, msgFEReady);
              ipc.server.stop();
            }
            onlineLocalSockets[obj.userid] = local_socket;
          });

          // sever get logic process's tcp package and transfer to FE
          // directly(not parse)
          ipc.server.on('data', function(buffer, local_socket) {
            ipc.log('got a data : ' + buffer.length);
            socket.emit('data', buffer);
          });

          ipc.server.on(
              'socket.disconnected', function(local_socket, destroyedSocketID) {
                ipc.log('client ' + destroyedSocketID + ' has disconnected!');
              });

        });
        ipc.server.start();
        onlineLocalIPC[obj.userid] = ipc;


        // Create BE process
        console.log(
            'path: ' +
            '/tmp/app.' + obj.username);
        // reader BE process path form config file
        var review_server_path;
        fs.readFile(
            __dirname + '/public/config/run_time.config', function(err, data) {
              if (err) {
                throw err;
              }
              review_server_path = data;
              console.log('app path : ' + review_server_path);

              // TODO Test 自己起C++ BE
              // const out_log = fs.openSync(
              //     './out_' + util.inspect(obj.username) + '.log', 'a');
              // const err_log = fs.openSync(
              //     './err_' + util.inspect(obj.username) + '.log', 'a');
              // var worker = childProcess.spawn(
              //     review_server_path.toString(), ['/tmp/app.' +
              //     obj.username],
              //     {detached: true, stdio: ['ignore', out_log, err_log]});
              // onlineLogicProcessID[obj.userid] = worker;
              // console.log('<><><><><><> login in success <><><><><><>');
            });
      });

      //监听用户退出
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

        // send last msg tp BE to notify it to shut down
        var ipc = onlineLocalIPC[userid];
        var localSocket = onlineLocalSockets[userid];
        // ipc.config.encoding='hex';

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
        delete onlineUsers[userid];
        delete onlineLocalSockets[userid];
        // delete onlineLogicProcessID[userid];
        delete onlineLocalIPC[userid];
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
    })



    var server = http.listen(8000, function() {
  var address = server.address();
  console.log('address is ', util.inspect(address));
});