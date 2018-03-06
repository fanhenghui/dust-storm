const childProcess = require('child_process');
const nodeIpc = require('node-ipc');
const config = require('../config/config');
const dbHandle = require('./db-handle');

const COMMAND_ID_BE_FE_SHUTDOWN = 120000;
const COMMAND_ID_BE_FE_READY = 120001;
const COMMAND_ID_BE_FE_HEARTBEAT = 119999;

class User {
    constructor(userID, userName) {
        this.userID = userID;
        this.userName = userName;
        this.ipcSocket = null;
        this.ipc = null;
        this.BEProcess = null;
        this.BETic = 0;
        this.FETic = 0;
        this.heartbeatInterval = null;
        this.onlineToken = '';
        this.onlineCheckingInterval = null;
        this.offlineFlag = 0;
    }
};

let onlineUserSet = {};
let onlineCount = 0;

const TIC_LIMIT = 4294967296;
const HEARTBEAT_INTERVAL = 5 * 1000;

function generateMsgBuffer(cmd_id) {
    const msgBuffer = new Buffer(32);
    msgBuffer.writeUIntLE(1, 0, 4);
    msgBuffer.writeUIntLE(0, 4, 4);
    msgBuffer.writeUIntLE(cmd_id, 8, 4);
    msgBuffer.writeUIntLE(0, 12, 4);
    msgBuffer.writeUIntLE(0, 16, 4);
    msgBuffer.writeUIntLE(0, 20, 4);
    msgBuffer.writeUIntLE(0, 24, 4);
    msgBuffer.writeUIntLE(0, 28, 4);
    return msgBuffer;
};

const HEARTBEAT_MSG_BUFFER = generateMsgBuffer(COMMAND_ID_BE_FE_HEARTBEAT);
const SHUTDOWN_BE_MSG_BUFFER = generateMsgBuffer(COMMAND_ID_BE_FE_SHUTDOWN);
const FE_READY_MSG_BUFFER = generateMsgBuffer(COMMAND_ID_BE_FE_READY);

function disconnectBE(userID) {
    ////注意 有极低的概率websocket调用disconnect失败

    if (!onlineUserSet.hasOwnProperty(userID)) {
        console.log(`${userID}: disconnecting failed.`);
        return;
    }

    //clear interval
    clearInterval(onlineUserSet[userID].heartbeatInterval);
    clearInterval(onlineUserSet[userID].onlineCheckingInterval);

    if (0 ==  onlineUserSet[userID].offlineFlag) {
        //正常退出登录的才清空online_token
        dbHandle.signOut(global.db, onlineUserSet[userID].userName);
    }
    
    let ipc = onlineUserSet[userID].ipc;
    let ipcSocket = onlineUserSet[userID].ipcSocket;
    let BEProcess = onlineUserSet[userID].BEProcess;
    
    if (ipc && ipcSocket && BEProcess) {
        //send last msg tp BE to notify BE shut down
        console.log(`${userID}: shut down BE.`);
        ipc.server.emit(ipcSocket, SHUTDOWN_BE_MSG_BUFFER);
        ipc.server.stop();
    }

    delete onlineUserSet[userID];
    onlineCount--;

    console.log(`${userID}: disconnecting success.`);

    //wait for kill worker(防止BE无法销毁)
    if (BEProcess) {
        setTimeout(()=>{
            console.log(`kill ${userID}'s process froce just in case`);
            BEProcess.kill('SIGHUP');
        }, 3000);
    }
};

function onlineChecking(socket, userID, userName, onlineToken) {
    global.db.query(`SELECT online_token FROM user WHERE name='${userName}'`, (err, data)=>{
        if (err) {
            console.log(`DB error: ${err}`);
        } else if (data.length != 0) {
            if (socket) {
                if (data[0].online_token != onlineToken) {
                    socket.emit('login_out','0');
                    onlineUserSet[userID].offlineFlag = 1;
                }
            }
        }
    });
}

module.exports = {
    onWebSocketConnect: function(socket) {
        socket.on('login', (obj)=>{
            const userName = obj.userName;
            const userID = obj.userID;
            const onlineToken = obj.onlineToken;

            if (!userName) {
                console.log('invalid userName!');
                return;
            }

            socket.name = userID; // uid to locate web socket
            console.log(`web-socket on: {userName: ${userName}, userID: ${userID}}`);

            if (!onlineUserSet.hasOwnProperty(userID)) {
                onlineUserSet[userID] = new User(userID, userName);
                onlineUserSet[userID].onlineToken = onlineToken;
                onlineCount++;
            } else {
                //Fatal error！
                console.log(userName + ' is online now.');
            }

            // create a server IPC between server and BE
            const ipc = new nodeIpc.IPC;
            onlineUserSet[userID].ipc = ipc;

            ipc.config.id = userID;
            ipc.config.retry = 1500;
            ipc.config.rawBuffer = true;
            ipc.config.encoding = 'hex';
            ipc.serve(()=>{
                ipc.server.on('connect', ipcSocket=>{
                    // when IPC is constructed between server and BE
                    // server send a ready MSG to BE to make sure BE is ready
                    console.log('IPC connect. Send FE ready to BE.');
                    onlineUserSet[userID].ipcSocket = ipcSocket;
                    if (ipc) {
                        ipc.server.emit(ipcSocket, FE_READY_MSG_BUFFER);
                    }

                    // send heart bet package
                    let heartbeatBE = function() {
                        if (onlineUserSet.hasOwnProperty(userID)) {                          
                            onlineUserSet[userID].ipc.server.emit(onlineUserSet[userID].ipcSocket, HEARTBEAT_MSG_BUFFER);

                            if (onlineUserSet[userID].BETic != onlineUserSet[userID].FETic) {
                                console.log(`${userID}: the heart does\'t jump more than  ${HEARTBEAT_INTERVAL}. kill BE.`);
                                disconnectBE(userID);
                            } else {
                                onlineUserSet[userID].BETic += 1;
                                if (onlineUserSet[userID].BETic > TIC_LIMIT) {
                                    onlineUserSet[userID].BETic = 0;
                                }
                                console.log(`${userID}: heartbeat: ${onlineUserSet[userID].BETic}`);
                            }
                        }
                    };
                    onlineUserSet[userID].BETic = 0;
                    onlineUserSet[userID].FETic = 0;
                    onlineUserSet[userID].heartbeatInterval = setInterval(heartbeatBE, HEARTBEAT_INTERVAL);
                });

                // sever get logic process's tcp package and transfer to FE directly(not parse)
                ipc.server.on('data', (buffer, ipcSocket)=> {
                    socket.emit('data', buffer);
                });

                ipc.server.on('socket.disconnected', (ipcSocket, destroyedSocketID)=> {
                    ipc.log('client ' + destroyedSocketID + ' has disconnected!');
                });
            });
            ipc.server.start();
            
            // Create BE process
            console.log(`path: /tmp/app.${userID}`);

            /// run process
            let worker = childProcess.spawn(config.be_path, ['/tmp/app.' + userID], {detached: true});
            onlineUserSet[userID].BEProcess = worker;

            //// std ouput to server device
            worker.stdout.on('data', data => {
                console.log(`stdout: ${data}`);
            });
            worker.stderr.on('data', data => {
                console.log(`stderr: ${data}`);
            });

            worker.on('close', code => {
                console.log(`child process exited with code ${code}`);
                if (onlineUserSet.hasOwnProperty(userID)) {
                    // BE主动断开：一般是BE crash
                    onlineUserSet[userID].BEProcess = null;
                    disconnectBE(userID);
                }
            });

            //online checking interval
            onlineUserSet[userID].onlineCheckingInterval = setInterval(function() {
                onlineChecking(socket, userID, userName, onlineToken);
            }, config.online_checking_interval);

            console.log(`${userID}: init done.`);
        });

        // 前台主动断开
        socket.on('disconnect', function(){
            const userID = socket.name;
            if (userID) {
                disconnectBE(userID);
            }
        });

        socket.on('heartbeat', obj => {
            const userID = obj.userID;
            if (onlineUserSet.hasOwnProperty(userID)) {
                onlineUserSet[userID].FETic += 1;
                if (onlineUserSet[userID].FETic > TIC_LIMIT) {
                    onlineUserSet[userID].FETic = 0;
                }
            }
        });

        socket.on('data', obj=>{
            //FE发来的TCP 粘包问题是在BE端解决(UNIX socket 可以控制每次recv的字节数，因此可以天然解决粘包问题)
            let userID = obj.userID;
            if (!onlineUserSet.hasOwnProperty(userID)) {
                console.log('socket on data , ERROR IPC');
                return;
            }
            let ipc = onlineUserSet[userID].ipc;
            let ipcSocket = onlineUserSet[userID].ipcSocket;

            let buffer = obj.content;
            ipc.server.emit(ipcSocket, buffer);

            //let commandID = buffer.readUIntLE(8, 4);
            //console.log('web send to server : userName ' + obj.userName);
            //console.log('buffer length : ' + buffer.byteLength);
            //console.log('command id :' + commandID);
        });
    },

    cleanWebSocketConnection: function() {
        for (let key in onlineUserSet) {
            disconnectBE(key);
        }
    }
}