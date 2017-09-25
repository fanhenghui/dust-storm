const MSG_HEADER_LEN = 32;

var SocketClient = {
  tcpPacketEnd:0,
  msgCmdID: 0,
  msgCellID: 0,
  msgOpID: 0,
  msgRestDataLen: 0,
  lastMsgHeader: new ArrayBuffer(MSG_HEADER_LEN),
  lastMsgHeaderLen: 0,
  protocRoot: null,

  recvData:function(tcpBuffer, msgHandler) {
    var tcpPackageLen = tcpBuffer.byteLength;
  
    if (tcpPacketEnd == 0) {
      if (tcpPackageLen < MSG_HEADER_LEN) {
        tcpPacketEnd = 2;
        lastMsgHeaderLen = tcpPackageLen;
        tcpBuffer.copy(lastMsgHeader, 0, 0);
        return;
      }
      var header = new Uint32Array(tcpBuffer, 0, 8);
      msgCmdID = header[2];
      msgCellID = header[3];
      msgOpID = header[4];
      var lastMsgDatalen = header[7];
  
      if (tcpPackageLen - MSG_HEADER_LEN == lastMsgDatalen) {  // completed one Msg
        msgHandler(msgCmdID, msgCellID, msgOpID, tcpBuffer, MSG_HEADER_LEN, lastMsgDatalen, 0, true);
      } else if (tcpPackageLen - MSG_HEADER_LEN < lastMsgDatalen) {  // not completed one Msg
        msgRestDataLen = lastMsgDatalen - (tcpPackageLen - MSG_HEADER_LEN);
        msgHandler(msgCmdID, msgCellID, msgOpID, tcpBuffer, MSG_HEADER_LEN, tcpPackageLen - 32, msgRestDataLen, true);
        tcpPacketEnd = 1;
      } else {  // this buffer carry next Msg process current one
        msgHandler(msgCmdID, msgCellID, msgOpID, tcpBuffer, MSG_HEADER_LEN, lastMsgDatalen, 0, true);
        // recursion process rest
        var tcpBufferSub = tcpBuffer.slice(lastMsgDatalen + MSG_HEADER_LEN);
        tcpPacketEnd = 0;
        recvData(tcpBufferSub);
      }
    } else if (tcpPacketEnd == 1) {// data for last msg
      if (tcpPackageLen - msgRestDataLen == 0) {  // complete last msg
        msgRestDataLen = 0;
        msgHandler(msgCmdID, msgCellID, msgOpID, tcpBuffer, 0, tcpPackageLen, 0, false);
      } else if (tcpPackageLen - msgRestDataLen < 0) {  // not complete data yet
        msgRestDataLen -= tcpPackageLen;
        tcpPacketEnd = 1;
        msgHandler(msgCmdID, msgCellID, msgOpID, tcpBuffer, 0, tcpPackageLen, msgRestDataLen, false);
      } else {  // this buffer carry next Msg
        msgHandler(msgCmdID, msgCellID, msgOpID, tcpBuffer, 0, msgRestDataLen, 0, false);
        var tcpBufferSub2 = tcpBuffer.slice(msgRestDataLen);
        msgRestDataLen = 0;
        tcpPacketEnd = 0;
        recvData(tcpBufferSub2);
      }
    } else if (tcpPacketEnd == 2) {  // msg header for last msg header
      var lastRestHeaderLen = MSG_HEADER_LEN - lastMsgHeaderLen;
      if (tcpPackageLen < lastRestHeaderLen) {  // msg header is not completed yet
        tcpPacketEnd = 2;
        tcpBuffer.copy(lastMsgHeader, 0, lastRestHeaderLen, tcpPackageLen);
        lastMsgHeaderLen += tcpPackageLen;
        return;
      } else {  // msg header is completed
        tcpPacketEnd = 1;
        tcpBuffer.copy(lastMsgHeader, 0, lastRestHeaderLen, lastRestHeaderLen);
        var tcpBufferSub3 = tcpBuffer.slice(lastRestHeaderLen);
  
        var header2 = new Uint32Array(lastMsgHeader, 0, 8);
        msgCmdID = header2[2];
        msgCellID = header2[3];
        msgOpID = header2[4];
        msgRestDataLen = header2[7];
  
        tcpPacketEnd = 1;
        lastMsgHeaderLen = 0;
        recvData(tcpBufferSub3);
      }
    }
  },

  sendData: function(socket, msgID, opID, cellID, dataLen, buffer) {
    var cmdBuffer = new ArrayBuffer(MSG_HEADER_LEN + dataLen);
    // header
    var header = new Uint32Array(cmdBuffer, 0, 8);
    header[0] = 0;
    header[1] = 0;
    header[2] = msgID;
    header[3] = cellID;
    header[4] = opID;
    header[5] = 0;
    header[6] = 0;
    header[7] = dataLen;
    // data
    var srcBuffer = new Uint8Array(buffer);
    var dstBuffer = new Uint8Array(cmdBuffer, 8 * 4, dataLen);
    for (var index = 0; index < dataLen; index++) {
      dstBuffer[index] = srcBuffer[index];
    }
    console.log('emit data.');
    socket.emit('data', {
      userid: socket.userID,
      username: socket.userName,
      content: cmdBuffer
    });
  },

  //heartbeat 
  hearbeat: function (socket) {
    socket.emit('heartbeat', {userid: socket.userID, username: socket.userName});
  },

  //load protoc
  loadProtoc: function (protoFile) {
    //load protocbuf file
    protobuf.load(protoFile, function(err, root) {
      if(err) {
        //TODO log
        console.log('load proto failed!');
        throw err;
      } else {
        protocRoot = root;
      }
    });
  },
};


//// TCP packet processing function
// var tcpPacketEnd = 0;  // 0 msg header 1 data for last msg 2 msg header for last msg header
// var msgCmdID = 0;
// var msgCellID = 0;
// var msgOpID = 0;
// var msgRestDataLen = 0;
// var lastMsgHeader = new ArrayBuffer(MSG_HEADER_LEN);
// var lastMsgHeaderLen = 0;

// var protocRoot = null;

// function recvData(tcpBuffer, msgHandler) {
//   var tcpPackageLen = tcpBuffer.byteLength;

//   if (tcpPacketEnd == 0) {
//     if (tcpPackageLen < MSG_HEADER_LEN) {
//       tcpPacketEnd = 2;
//       lastMsgHeaderLen = tcpPackageLen;
//       tcpBuffer.copy(lastMsgHeader, 0, 0);
//       return;
//     }
//     var header = new Uint32Array(tcpBuffer, 0, 8);
//     msgCmdID = header[2];
//     msgCellID = header[3];
//     msgOpID = header[4];
//     var lastMsgDatalen = header[7];

//     if (tcpPackageLen - MSG_HEADER_LEN == lastMsgDatalen) {  // completed one Msg
//       msgHandler(msgCmdID, msgCellID, msgOpID, tcpBuffer, MSG_HEADER_LEN, lastMsgDatalen, 0, true);
//     } else if (tcpPackageLen - MSG_HEADER_LEN < lastMsgDatalen) {  // not completed one Msg
//       msgRestDataLen = lastMsgDatalen - (tcpPackageLen - MSG_HEADER_LEN);
//       msgHandler(msgCmdID, msgCellID, msgOpID, tcpBuffer, MSG_HEADER_LEN, tcpPackageLen - 32, msgRestDataLen, true);
//       tcpPacketEnd = 1;
//     } else {  // this buffer carry next Msg process current one
//       msgHandler(msgCmdID, msgCellID, msgOpID, tcpBuffer, MSG_HEADER_LEN, lastMsgDatalen, 0, true);
//       // recursion process rest
//       var tcpBufferSub = tcpBuffer.slice(lastMsgDatalen + MSG_HEADER_LEN);
//       tcpPacketEnd = 0;
//       recvData(tcpBufferSub);
//     }
//   } else if (tcpPacketEnd == 1) {// data for last msg
//     if (tcpPackageLen - msgRestDataLen == 0) {  // complete last msg
//       msgRestDataLen = 0;
//       msgHandler(msgCmdID, msgCellID, msgOpID, tcpBuffer, 0, tcpPackageLen, 0, false);
//     } else if (tcpPackageLen - msgRestDataLen < 0) {  // not complete data yet
//       msgRestDataLen -= tcpPackageLen;
//       tcpPacketEnd = 1;
//       msgHandler(msgCmdID, msgCellID, msgOpID, tcpBuffer, 0, tcpPackageLen, msgRestDataLen, false);
//     } else {  // this buffer carry next Msg
//       msgHandler(msgCmdID, msgCellID, msgOpID, tcpBuffer, 0, msgRestDataLen, 0, false);
//       var tcpBufferSub2 = tcpBuffer.slice(msgRestDataLen);
//       msgRestDataLen = 0;
//       tcpPacketEnd = 0;
//       recvData(tcpBufferSub2);
//     }
//   } else if (tcpPacketEnd == 2) {  // msg header for last msg header
//     var lastRestHeaderLen = MSG_HEADER_LEN - lastMsgHeaderLen;
//     if (tcpPackageLen < lastRestHeaderLen) {  // msg header is not completed yet
//       tcpPacketEnd = 2;
//       tcpBuffer.copy(lastMsgHeader, 0, lastRestHeaderLen, tcpPackageLen);
//       lastMsgHeaderLen += tcpPackageLen;
//       return;
//     } else {  // msg header is completed
//       tcpPacketEnd = 1;
//       tcpBuffer.copy(lastMsgHeader, 0, lastRestHeaderLen, lastRestHeaderLen);
//       var tcpBufferSub3 = tcpBuffer.slice(lastRestHeaderLen);

//       var header2 = new Uint32Array(lastMsgHeader, 0, 8);
//       msgCmdID = header2[2];
//       msgCellID = header2[3];
//       msgOpID = header2[4];
//       msgRestDataLen = header2[7];

//       tcpPacketEnd = 1;
//       lastMsgHeaderLen = 0;
//       recvData(tcpBufferSub3);
//     }
//   }
// }

// //need socket add userID&userName attribute
// function sendData(socket, msgID, opID, cellID, dataLen, buffer) {
//   var cmdBuffer = new ArrayBuffer(MSG_HEADER_LEN + dataLen);
//   // header
//   var header = new Uint32Array(cmdBuffer, 0, 8);
//   header[0] = 0;
//   header[1] = 0;
//   header[2] = msgID;
//   header[3] = cellID;
//   header[4] = opID;
//   header[5] = 0;
//   header[6] = 0;
//   header[7] = dataLen;
//   // data
//   var srcBuffer = new Uint8Array(buffer);
//   var dstBuffer = new Uint8Array(cmdBuffer, 8 * 4, dataLen);
//   for (var index = 0; index < dataLen; index++) {
//     dstBuffer[index] = srcBuffer[index];
//   }
//   console.log('emit data.');
//   socket.emit('data', {
//     userid: socket.userID,
//     username: socket.userName,
//     content: cmdBuffer
//   });
// }

// //heartbeat 
// function hearbeat(socket) {
//   socket.emit('heartbeat', {userid: socket.userID, username: socket.userName});
// }

// //load protoc
// function loadProtoc(protoFile) {
//   //load protocbuf file
//   protobuf.load(protoFile, function(err, root) {
//     if(err) {
//         //TODO log
//         console.log('load proto failed!');
//         throw err;
//     } else {
//         protocRoot = root;
//     }
// });
// }