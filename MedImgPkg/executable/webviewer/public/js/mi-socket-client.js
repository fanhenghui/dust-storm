const MSG_HEADER_LEN = 32;

function SocketClient(socket) {
  this.tcpPacketEnd = 0;
  this.msgCmdID = 0;
  this.msgCellID = 0;
  this.msgOpID = 0;
  this.msgRestDataLen = 0;
  this.lastMsgHeader = new ArrayBuffer(MSG_HEADER_LEN);
  this.lastMsgHeaderLen = 0;
  this.protocRoot = null;
  this.socket = socket;
}

//Handler(cmdID, cellID, opID, tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader)
SocketClient.prototype.recvData = function(tcpBuffer, msgHandler) {
  var tcpPackageLen = tcpBuffer.byteLength;

  if (this.tcpPacketEnd == 0) {
    if (tcpPackageLen < MSG_HEADER_LEN) {
      this.tcpPacketEnd = 2;
      this.lastMsgHeaderLen = tcpPackageLen;
      tcpBuffer.copy(this.lastMsgHeader, 0, 0);
      return;
    }
    var header = new Uint32Array(tcpBuffer, 0, 8);
    this.msgCmdID = header[2];
    this.msgCellID = header[3];
    this.msgOpID = header[4];
    var lastMsgDatalen = header[7];

    if (tcpPackageLen - MSG_HEADER_LEN == lastMsgDatalen) {  // completed one Msg
      msgHandler(this.msgCmdID, this.msgCellID, this.msgOpID, tcpBuffer, MSG_HEADER_LEN, lastMsgDatalen, 0, true);
    } else if (tcpPackageLen - MSG_HEADER_LEN < lastMsgDatalen) {  // not completed one Msg
      this.msgRestDataLen = lastMsgDatalen - (tcpPackageLen - MSG_HEADER_LEN);
      msgHandler(this.msgCmdID, this.msgCellID, this.msgOpID, tcpBuffer, MSG_HEADER_LEN, tcpPackageLen - MSG_HEADER_LEN, this.msgRestDataLen, true);
      this.tcpPacketEnd = 1;
    } else {  // this buffer carry next Msg process current one
      msgHandler(this.msgCmdID, this.msgCellID, this.msgOpID, tcpBuffer, MSG_HEADER_LEN, lastMsgDatalen, 0, true);
      // recursion process rest
      var tcpBufferSub = tcpBuffer.slice(lastMsgDatalen + MSG_HEADER_LEN);
      this.tcpPacketEnd = 0;
      this.recvData(tcpBufferSub, msgHandler);
    }
  } else if (this.tcpPacketEnd == 1) {// data for last msg
    if (tcpPackageLen - this.msgRestDataLen == 0) {  // complete last msg
      this.msgRestDataLen = 0;
      msgHandler(this.msgCmdID, this.msgCellID, this.msgOpID, tcpBuffer, 0, tcpPackageLen, 0, false);
    } else if (tcpPackageLen - this.msgRestDataLen < 0) {  // not complete data yet
      this.msgRestDataLen -= tcpPackageLen;
      this.tcpPacketEnd = 1;
      msgHandler(this.msgCmdID, this.msgCellID, this.msgOpID, tcpBuffer, 0, tcpPackageLen, this.msgRestDataLen, false);
    } else {  // this buffer carry next Msg
      msgHandler(this.msgCmdID, this.msgCellID, this.msgOpID, tcpBuffer, 0, this.msgRestDataLen, 0, false);
      var tcpBufferSub2 = tcpBuffer.slice(this.msgRestDataLen);
      this.msgRestDataLen = 0;
      this.tcpPacketEnd = 0;
      this.recvData(tcpBufferSub2, msgHandler);
    }
  } else if (this.tcpPacketEnd == 2) {  // msg header for last msg header
    var lastRestHeaderLen = MSG_HEADER_LEN - this.lastMsgHeaderLen;
    if (tcpPackageLen < lastRestHeaderLen) {  // msg header is not completed yet
      this.tcpPacketEnd = 2;
      tcpBuffer.copy(this.lastMsgHeader, 0, lastRestHeaderLen, tcpPackageLen);
      this.lastMsgHeaderLen += tcpPackageLen;
      return;
    } else {  // msg header is completed
      this.tcpPacketEnd = 1;
      tcpBuffer.copy(this.lastMsgHeader, 0, lastRestHeaderLen, lastRestHeaderLen);
      var tcpBufferSub3 = tcpBuffer.slice(lastRestHeaderLen);

      var header2 = new Uint32Array(this.lastMsgHeader, 0, 8);
      this.msgCmdID = header2[2];
      this.msgCellID = header2[3];
      this.msgOpID = header2[4];
      this.msgRestDataLen = header2[7];

      this.tcpPacketEnd = 1;
      this.lastMsgHeaderLen = 0;
      this.recvData(tcpBufferSub3, msgHandler);
    }
  }
}

SocketClient.prototype.sendData = function(msgID, opID, cellID, dataLen, buffer) {
  if(dataLen <= 0) {
    let headerBuffer = new ArrayBuffer(MSG_HEADER_LEN);
    let header = new Uint32Array(headerBuffer);
    header[0] = 0;
    header[1] = 0;
    header[2] = msgID;
    header[3] = cellID;
    header[4] = opID;
    header[5] = 0;
    header[6] = 0;
    header[7] = 0;

    console.log('emit data length 0.');
    this.socket.emit('data', {
      userid: this.socket.userID,
      username: this.socket.userName,
      content: headerBuffer
    });
  } else {
    let cmdBuffer = new ArrayBuffer(MSG_HEADER_LEN + dataLen);
    // header
    let header = new Uint32Array(cmdBuffer, 0, 8);
    header[0] = 0;
    header[1] = 0;
    header[2] = msgID;
    header[3] = cellID;
    header[4] = opID;
    header[5] = 0;
    header[6] = 0;
    header[7] = dataLen;
    // data
    let srcBuffer = new Uint8Array(buffer);
    let dstBuffer = new Uint8Array(cmdBuffer, 8 * 4, dataLen);
    for (let index = 0; index < dataLen; index++) {
      dstBuffer[index] = srcBuffer[index];
    }
    console.log('emit data length' + dataLen + '.');
    this.socket.emit('data', {
      userid: this.socket.userID,
      username: this.socket.userName,
      content: cmdBuffer
    });
  }
}

//heartbeat 
SocketClient.prototype.heartbeat = function () {
  this.socket.emit('heartbeat', {userid: this.socket.userID, username: this.socket.userName});
}

//load protoc
SocketClient.prototype.loadProtoc = function (protoFile) {
  //load protocbuf file
  var loadProtoc_ = (function(err, root) {
    if(err) {
      //TODO log
      console.log('load proto failed!');
      throw err;
    } else {
      this.protocRoot = root;
    }
  }).bind(this);
  protobuf.load(protoFile, loadProtoc_);
}