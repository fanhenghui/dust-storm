const MSG_HEADER_LEN = 32;

class SocketClient {
    constructor(socket) {
        this._tcpPacketEnd = 0;
        this._msgCmdID = 0;
        this._msgCellID = 0;
        this._msgOpID = 0;
        this._msgRestDataLen = 0;
        this._lastMsgHeader = new ArrayBuffer(MSG_HEADER_LEN);
        this._lastMsgHeaderLen = 0;
        this._socket = socket;
        
        this.protocRoot = null;
    }

    //Handler(cmdID, cellID, opID, tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader)
    recvData(tcpBuffer, msgHandler) {
        const tcpPackageLen = tcpBuffer.byteLength;
    
        if (this._tcpPacketEnd == 0) {
            if (tcpPackageLen < MSG_HEADER_LEN) { //incompleted Msg header
                let dstBuffer = new Uint8Array(this._lastMsgHeader);
                let srcBuffer = new Uint8Array(tcpBuffer)
                for (let i = 0; i< tcpPackageLen; ++i) {
                    dstBuffer[i] = srcBuffer[i];
                }
                this._tcpPacketEnd = 2;
                this._lastMsgHeaderLen = tcpPackageLen;
                return;
            }
            let header = new Uint32Array(tcpBuffer, 0, 8);
            this._msgCmdID = header[2];
            this._msgCellID = header[3];
            this._msgOpID = header[4];
            const lastMsgDatalen = header[7];
    
            if (tcpPackageLen - MSG_HEADER_LEN == lastMsgDatalen) { // completed one Msg
                msgHandler(this._msgCmdID, this._msgCellID, this._msgOpID, tcpBuffer, MSG_HEADER_LEN, lastMsgDatalen, 0, true);
            } else if (tcpPackageLen - MSG_HEADER_LEN < lastMsgDatalen) { // not completed one Msg
                this._msgRestDataLen = lastMsgDatalen - (tcpPackageLen - MSG_HEADER_LEN);
                msgHandler(this._msgCmdID, this._msgCellID, this._msgOpID, tcpBuffer, MSG_HEADER_LEN, tcpPackageLen - MSG_HEADER_LEN, this._msgRestDataLen, true);
                this._tcpPacketEnd = 1;
            } else { // this buffer carry next Msg process current one
                msgHandler(this._msgCmdID, this._msgCellID, this._msgOpID, tcpBuffer, MSG_HEADER_LEN, lastMsgDatalen, 0, true);
                // recursion process rest
                let tcpBufferSub = tcpBuffer.slice(lastMsgDatalen + MSG_HEADER_LEN);
                this._tcpPacketEnd = 0;
                this.recvData(tcpBufferSub, msgHandler);
            }
        } else if (this._tcpPacketEnd == 1) { // data for last msg
            if (tcpPackageLen - this._msgRestDataLen == 0) { // complete last msg
                this._msgRestDataLen = 0;
                msgHandler(this._msgCmdID, this._msgCellID, this._msgOpID, tcpBuffer, 0, tcpPackageLen, 0, false);
            } else if (tcpPackageLen - this._msgRestDataLen < 0) { // not complete data yet
                this._msgRestDataLen -= tcpPackageLen;
                this._tcpPacketEnd = 1;
                msgHandler(this._msgCmdID, this._msgCellID, this._msgOpID, tcpBuffer, 0, tcpPackageLen, this._msgRestDataLen, false);
            } else { // this buffer carry next Msg
                msgHandler(this._msgCmdID, this._msgCellID, this._msgOpID, tcpBuffer, 0, this._msgRestDataLen, 0, false);
                let tcpBufferSub2 = tcpBuffer.slice(this._msgRestDataLen);
                this._msgRestDataLen = 0;
                this._tcpPacketEnd = 0;
                this.recvData(tcpBufferSub2, msgHandler);
            }
        } else if (this._tcpPacketEnd == 2) { // msg header for last msg header
            const lastRestHeaderLen = MSG_HEADER_LEN - this._lastMsgHeaderLen;
            if (tcpPackageLen < lastRestHeaderLen) { // msg header is not completed yet
                let dstBuffer = new Uint8Array(this._lastMsgHeader);
                let srcBuffer = new Uint8Array(tcpBuffer)
                for (let i = 0 ; i< tcpPackageLen; ++i) {
                    dstBuffer[i+this._lastMsgHeaderLen] = srcBuffer[i];
                }
                this._tcpPacketEnd = 2;
                this._lastMsgHeaderLen += tcpPackageLen;
                return;
            } else { // msg header is completed
                //fill header completed
                let dstBuffer = new Uint8Array(this._lastMsgHeader);
                let srcBuffer = new Uint8Array(tcpBuffer,0,lastRestHeaderLen);
                for (let i = 0; i< lastRestHeaderLen; ++i) {
                    dstBuffer[i+this._lastMsgHeaderLen] = srcBuffer[i];
                }
    
                let tcpBufferSub3 = tcpBuffer.slice(lastRestHeaderLen);
                let header2 = new Uint32Array(this._lastMsgHeader, 0, 8);
                this._msgCmdID = header2[2];
                this._msgCellID = header2[3];
                this._msgOpID = header2[4];
                this._msgRestDataLen = header2[7];
    
                this._tcpPacketEnd = 1;
                this._lastMsgHeaderLen = 0;
                this.recvData(tcpBufferSub3, msgHandler);
            }
        }
    }

    sendData(msgID, opID, cellID, dataLen, buffer) {
        if (dataLen <= 0) {
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
    
            //console.log('emit data length 0.');
            this._socket.emit('data', {
                userid: this._socket.userID,
                username: this._socket.userName,
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
            //console.log('emit data length' + dataLen + '.');
            this._socket.emit('data', {
                userid: this._socket.userID,
                username: this._socket.userName,
                content: cmdBuffer
            });
        }
    }

    //heartbeat 
    heartbeat() {
        this._socket.emit('heartbeat', {
            userid: this._socket.userID,
            username: this._socket.userName
        });
    }

    //load protoc
    loadProtoc(protoFile) {
        //load protocbuf file
        let loadProtoc_ = (function(err, root) {
            if (err) {
                console.log('load proto failed!');
            } else {
                this.protocRoot = root;
            }
        }).bind(this);
        protobuf.load(protoFile, loadProtoc_);
    }
}