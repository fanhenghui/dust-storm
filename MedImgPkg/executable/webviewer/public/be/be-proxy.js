const fs = require('fs');
const childProcess = require('child_process');

const COMMAND_ID_FE_SHUT_DOWN = 120000;
const COMMAND_ID_FE_READY = 120001;
const COMMAND_ID_FE_HEARTBEAT = 119999;

let onlineUsers = {};
let onlineLocalSockets = {};
let onlineLocalIPC = {};
let onlineLogicProcess = {};
let onlineBEHBClock = {};
let onlineBETic = {};
let onlineFETic = {};
let onlineCount = 0;

const TIC_LIMIT = 4294967296;
const HEARTBEAT_INTERVAL = 5 * 1000;

let disconnectBE = (userid)=>{
    if (!onlineUsers.hasOwnProperty(userid)) {
        console.log(userid + ' disconnecting failed.');
        return;
    }

    console.log(userid + ' has disconnecting.');

    // send last msg tp BE to notify BE shut down
    let ipc = onlineLocalIPC.hasOwnProperty(userid) ? onlineLocalIPC[userid] : null;
    let localSocket = onlineLocalSockets.hasOwnProperty(userid) ? onlineLocalSockets[userid] : null;

    let quitMsg = new Buffer(32);
    quitMsg.writeUIntLE(1, 0, 4);
    quitMsg.writeUIntLE(0, 4, 4);
    quitMsg.writeUIntLE(COMMAND_ID_FE_SHUT_DOWN, 8, 4);
    quitMsg.writeUIntLE(0, 12, 4);
    quitMsg.writeUIntLE(0, 16, 4);
    quitMsg.writeUIntLE(0, 20, 4);
    quitMsg.writeUIntLE(0, 24, 4);
    quitMsg.writeUIntLE(0, 28, 4);

    if (ipc) {
        ipc.server.emit(localSocket, quitMsg);
        ipc.server.stop();
    }

    // purge user's infos
    if (onlineUsers.hasOwnProperty(userid)) {
        let username = onlineUsers[userid];
        //clear user online info (DB)
        //注意 有极低的概率websocket调用disconnect失败
        global.dbHandel.signOut(username);
        delete onlineUsers[userid];
    }
    if (onlineLocalSockets.hasOwnProperty(userid)) {
        delete onlineLocalSockets[userid];
    }
    if (onlineLocalIPC.hasOwnProperty(userid)) {
        delete onlineLocalIPC[userid];
    }
    if (onlineBEHBClock.hasOwnProperty(userid)) {
        clearInterval(onlineBEHBClock[userid]);
        delete onlineBEHBClock[userid];
    }
    onlineCount--;
    console.log(userid + ' disconnecting success.');

    //wait for kill worker
    setTimeout(()=>{
        if (onlineLogicProcess.hasOwnProperty(userid)) {
            console.log('kill ' + userid + '\'s process froce just in case');
            onlineLogicProcess[userid].kill('SIGHUP');
            delete onlineLogicProcess[userid];
        }
    }, 3000);
};

module.exports = {
    onWebSocketConnect: function(socket) {
        console.log('<><><><><><> connecting <><><><><><>');
        //监听用户加入
        socket.on('login', (obj)=>{
            if (obj.username == '') {
                console.log('invalid null username!');
                return;
            }

            socket.name = obj.userid; // uid to locate web socket
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
            const nodeIpc = require('node-ipc');
            const ipc = new nodeIpc.IPC;

            console.log('UNIX path is : ' + obj.username);
            ipc.config.id = obj.username;
            ipc.config.retry = 1500;
            ipc.config.rawBuffer = true;
            ipc.config.encoding = 'hex';
            ipc.serve(()=>{
                ipc.server.on('connect', (localSocket)=>{
                    // when IPC is constructed between server and BE
                    // server send a ready MSG to BE to make sure BE is ready
                    console.log('IPC connect. Send FE ready to BE.');
                    onlineLocalSockets[obj.userid] = localSocket;

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

                    // send heart bet package
                    let heartbeatBE = ()=>{
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

                            //check heartbeat
                            if (onlineBETic[obj.userid] != onlineFETic[obj.userid]) {
                                //TODO disconnect
                                console.log('the heart does\'t jump more than ' + HEARTBEAT_INTERVAL + '. kill BE.');
                                disconnectBE();
                            }
                            onlineBETic[obj.userid] += 1;
                            if (onlineBETic[obj.userid] > TIC_LIMIT) {
                                onlineBETic[obj.userid] = 0;
                            }
                            console.log('server heart beat for user: ' + obj.username + ' ' + onlineBETic[obj.userid]);
                        }
                    };
                    onlineBETic[obj.userid] = 0;
                    onlineFETic[obj.userid] = 0;
                    onlineBEHBClock[obj.userid] = setInterval(heartbeatBE, HEARTBEAT_INTERVAL);
                });

                // sever get logic process's tcp package and transfer to FE directly(not parse)
                ipc.server.on('data', (buffer, localSocket)=> {
                    //ipc.log('got a data : ' + buffer.length);
                    socket.emit('data', buffer);
                });

                ipc.server.on('socket.disconnected', (localSocket, destroyedSocketID)=> {
                    ipc.log('client ' + destroyedSocketID + ' has disconnected!');
                });
            });

            ipc.server.start();
            onlineLocalIPC[obj.userid] = ipc;

            // Create BE process
            console.log('path: ' + '/tmp/app.' + obj.username);
            // reader BE process path form config file
            fs.readFile(__dirname + '/be_path', (err, data)=>{
                if (err) {
                    console.log('null be path!');
                    throw err;
                }
                //read line 0 as be path
                let lines = data.toString().split('\n');
                if (lines.length == 0) {
                    console.log('read be path fiailed!');
                    throw('read be path fiailed!');
                }
                let review_server_path = lines[0];
                console.log('app path : ' + review_server_path);

                /// run process
                let worker = childProcess.spawn(review_server_path, ['/tmp/app.' + obj.username], {
                    detached: true
                });
                onlineLogicProcess[obj.userid] = worker;

                //// std ouput to server device
                worker.stdout.on('data', (data) => {
                    console.log(`stdout: ${data}`);
                });
                worker.stderr.on('data', (data) => {
                    console.log(`stderr: ${data}`);
                });

                worker.on('close', (code) => {
                    console.log('child process exited with code: ' + code);
                    //TODO 检查是否是正常退出，如果不是，而且websocket还在连接中，通知FE BE crash, 并断开连接
                });
                console.log('<><><><><><> create review server success <><><><><><>');
            });
        });

        let disconnectBEExt = ()=>{
            let userid = socket.name;
            disconnectBE(userid);
        };

        // listen user disconnect(leave review page)
        socket.on('disconnect', disconnectBEExt);

        socket.on('heartbeat', (obj) => {
            if (onlineFETic.hasOwnProperty(obj.userid)) {
                onlineFETic[obj.userid] += 1;
                if (onlineFETic[obj.userid] > TIC_LIMIT) {
                    onlineFETic[obj.userid] = 0;
                }
                //console.log('server receive FE heart beat for user: ' + obj.username + ' ' + onlineFETic[obj.userid]);
            }
        });

        socket.on('data', (obj)=>{
            //FE发来的TCP 粘包问题是在BE端解决(UNIX socket 可以控制每次recv的字节数，因此可以天然解决粘包问题)
            let userid = obj.userid;
            //console.log('socket on data , userid : ' + userid);
            if (!onlineLocalIPC.hasOwnProperty(userid)) {
                console.log('socket on data , ERROR IPC');
                return;
            }
            let ipc = onlineLocalIPC[userid];
            if (!onlineLocalSockets.hasOwnProperty(userid)) {
                console.log('socket on data , ERROR SOCKET');
                reutrn;
            }
            let localSocket = onlineLocalSockets[userid];

            let buffer = obj.content;
            let commandID = buffer.readUIntLE(8, 4);
            //console.log('web send to server : username ' + obj.username);
            //console.log('buffer length : ' + buffer.byteLength);
            //console.log('command id :' + commandID);

            ipc.server.emit(localSocket, buffer);
        });
    },

    cleanWebSocketConnection: function() {
        let innerUsers = {};
        for (let userid in onlineUsers) {
            innerUsers[userid] = onlineUsers[userid];
        }
        for (let userid in innerUsers) {
            disconnectBE(userid);
        }
    }
}