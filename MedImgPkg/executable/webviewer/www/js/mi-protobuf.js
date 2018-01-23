class Protobuf {
    static encode(socketClient, msgType, data) {
        if (!socketClient) {
            console.log('socket client is null.')
            return null;
        }
        let MsgType = socketClient.protocRoot.lookup(`medical_imaging.${msgType}`);
        if (!MsgType) {
            console.log(`get ${msgType} type failed.`);
            return null;
        }

        let buffer = null;
        try {
            let msg = MsgType.create(data);
            if (!msg) {
                console.log(`create ${msgType} failed with data: ${data}.`);
                return null;
            }
    
            buffer = MsgType.encode(msg).finish();
            if (!buffer) {
                console.log(`encode ${msgType} failed with data: ${data}.`);
                return null;
            }
            if (buffer.byteLength == 0) {
                console.log(`encode ${msgType} failed with data: ${data}. 0 bytelength.`);
                return null;
            }
        } catch(e) {
            if (e instanceof protobuf.util.ProtocolError) {
                console.log(`exception ProtocolError: encode ${msgType} failed with data: ${data}.`);
            } else {
                console.log(`exception: encode ${msgType} failed with data: ${data}.`);
            }
            return null;
        }
        
        return buffer;
    }

    static decode(socketClient, msgType, buffer) {
        if (!socketClient) {
            console.log('socket client is null.')
            return null;
        }
        
        if (!buffer) {
            console.log(`can't decode ${msgType} from null buffer.`);
            return null;
        }

        let MsgType = socketClient.protocRoot.lookup(`medical_imaging.${msgType}`);
        if (!MsgType) {
            console.log(`get ${msgType} type failed.`);
            return null;
        }
        
        let msg = null;
        try {
            let buffer8Uint = new Uint8Array(buffer);
            msg = MsgType.decode(buffer8Uint);
        } catch(e) {
            if (e instanceof protobuf.util.ProtocolError) {
                console.log(`exception ProtocolError: decode ${msgType} failed.`);
                console.log('e.instance holds the so far decoded message with missing required fields');
            } else {
                console.log(`exception: decode ${msgType} failed.`);
            }
            return null;
        }
        return msg;
    }
};