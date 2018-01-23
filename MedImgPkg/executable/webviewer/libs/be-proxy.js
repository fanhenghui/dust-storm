const childProcess = require('child_process');
const nodeIpc = require('node-ipc');
const config = require('../config/config');
const dbHandle = require('./db-handle');

const COMMAND_ID_BE_FE_SHUTDOWN = 120000;
const COMMAND_ID_BE_FE_READY = 120001;
const COMMAND_ID_BE_FE_HEARTBEAT = 119999;

class User {
    constructor(userid, username) {
        this.userid = userid;
        this.username = username;
        this.ipcSocket = null;
        this.ipc = null;
        this.BEProcess = null;
        this.BETic = 0;
        this.FETic = 0;
        this.heartbeatInterval = null;
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

let disconnectBE = (userid)=>{
    if (!onlineUserSet.hasOwnProperty(userid)) {
        console.log(userid + ' disconnecting failed.');
        return;
    }

    console.log(userid + ' has disconnecting.');

    // send last msg tp BE to notify BE shut down
    const username = onlineUserSet[userid].username;
    //clear user online info (DB)
    //注意 有极低的概率websocket调用disconnect失败
    dbHandle.signOut(global.db, username);

    let ipc = onlineUserSet[userid].ipc;
    let ipcSocket = onlineUserSet[userid].ipcSocket;
    let BEProcess = onlineUserSet[userid].BEProcess;
    
    if (ipc && ipcSocket && BEProcess) {
        console.log(userid + ' shut down BE.');
        ipc.server.emit(ipcSocket, SHUTDOWN_BE_MSG_BUFFER);
        ipc.server.stop();
    }
    
    clearInterval(onlineUserSet[userid].heartbeatInterval);
    delete onlineUserSet[userid];
    onlineCount--;

    console.log(userid + ' disconnecting success.');

    //wait for kill worker(防止BE无法销毁)
    if (BEProcess) {
        setTimeout(()=>{
            console.log('kill ' + userid + '\'s process froce just in case');
            BEProcess.kill('SIGHUP');
        }, 3000);
    }
};

module.exports = {
    onWebSocketConnect: function(socket) {
        console.log('<><><><><><> connecting <><><><><><>');
        //监听用户加入
        socket.on('login', (obj)=>{
            const username = obj.username;
            const userid = obj.userid;

            if (!username) {
                console.log('invalid username!');
                return;
            }

            socket.name = userid; // uid to locate web socket
            console.log('<><><><><><> logining <><><><><><>');
            console.log(username + ' has login.');
            console.log('id : ' + userid);

            if (!onlineUserSet.hasOwnProperty(userid)) {
                onlineUserSet[userid] = new User(userid, username);
                onlineCount++;
            } else {
                console.log(username + ' is online now.');
            }

            // create a server IPC between server and BE
            const ipc = new nodeIpc.IPC;
            onlineUserSet[userid].ipc = ipc;

            console.log(`UNIX path is : ${username}.`);
            ipc.config.id = username;
            ipc.config.retry = 1500;
            ipc.config.rawBuffer = true;
            ipc.config.encoding = 'hex';
            ipc.serve(()=>{
                ipc.server.on('connect', (ipcSocket)=>{
                    // when IPC is constructed between server and BE
                    // server send a ready MSG to BE to make sure BE is ready
                    console.log('IPC connect. Send FE ready to BE.');
                    onlineUserSet[userid].ipcSocket = ipcSocket;

                    //onlineLocalSockets[userid] = ipcSocket;
                    if (ipc) {
                        ipc.server.emit(ipcSocket, FE_READY_MSG_BUFFER);
                    }

                    // send heart bet package
                    let heartbeatBE = function() {
                        if (onlineUserSet.hasOwnProperty(userid)) {                          
                            onlineUserSet[userid].ipc.server.emit(onlineUserSet[userid].ipcSocket, HEARTBEAT_MSG_BUFFER);
                            //check heartbeat
                            if (onlineUserSet[userid].BETic != onlineUserSet[userid].FETic) {
                                console.log('the heart does\'t jump more than ' + HEARTBEAT_INTERVAL + '. kill BE.');
                                disconnectBE();
                            }
                            onlineUserSet[userid].BETic += 1;
                            if (onlineUserSet[userid].BETic > TIC_LIMIT) {
                                onlineUserSet[userid].BETic = 0;
                            }
                            console.log('server heart beat for user: ' + username + ' ' + onlineUserSet[userid].BETic);
                        }
                    };
                    onlineUserSet[userid].BETic = 0;
                    onlineUserSet[userid].FETic = 0;
                    onlineUserSet[userid].heartbeatInterval = setInterval(heartbeatBE, HEARTBEAT_INTERVAL);
                });

                // sever get logic process's tcp package and transfer to FE directly(not parse)
                ipc.server.on('data', (buffer, ipcSocket)=> {
                    //ipc.log('got a data : ' + buffer.length);
                    socket.emit('data', buffer);
                });

                ipc.server.on('socket.disconnected', (ipcSocket, destroyedSocketID)=> {
                    ipc.log('client ' + destroyedSocketID + ' has disconnected!');
                });
            });
            ipc.server.start();
            

            // Create BE process
            console.log('path: ' + '/tmp/app.' + obj.username);
            // reader BE process path form config file
            console.log('BE path: ' +  config.be_path);
            
            /// run process
            let worker = childProcess.spawn(config.be_path, ['/tmp/app.' + obj.username], {detached: true});
            onlineUserSet[userid].BEProcess = worker;

            //// std ouput to server device
            worker.stdout.on('data', (data) => {
                console.log(`stdout: ${data}`);
            });
            worker.stderr.on('data', (data) => {
                console.log(`stderr: ${data}`);
            });

            worker.on('close', (code) => {
                console.log('child process exited with code: ' + code);
                if (onlineUserSet.hasOwnProperty(userid)) {
                    // 后台主动断开：一般是crash了
                    onlineUserSet[userid].BEProcess = null;
                    disconnectBE(userid);
                }
                //TODO 检查是否是正常退出，如果不是，而且websocket还在连接中，通知FE BE crash, 并断开连接
            });

            console.log('<><><><><><> create review server success <><><><><><>');
        });

        // 前台主动断开
        socket.on('disconnect', ()=>{
            const userid = socket.name;
            if (userid) {
                disconnectBE(userid);
            }
        });

        socket.on('heartbeat', (obj) => {
            const userid = obj.userid;
            if (onlineUserSet.hasOwnProperty(userid)) {
                onlineUserSet[userid].FETic += 1;
                if (onlineUserSet[userid].FETic > TIC_LIMIT) {
                    onlineUserSet[userid].FETic = 0;
                }
            }
        });

        socket.on('data', (obj)=>{
            //FE发来的TCP 粘包问题是在BE端解决(UNIX socket 可以控制每次recv的字节数，因此可以天然解决粘包问题)
            let userid = obj.userid;
            if (!onlineUserSet.hasOwnProperty(userid)) {
                console.log('socket on data , ERROR IPC');
                return;
            }
            let ipc = onlineUserSet[userid].ipc;
            let ipcSocket = onlineUserSet[userid].ipcSocket;

            let buffer = obj.content;
            ipc.server.emit(ipcSocket, buffer);

            //let commandID = buffer.readUIntLE(8, 4);
            //console.log('web send to server : username ' + obj.username);
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