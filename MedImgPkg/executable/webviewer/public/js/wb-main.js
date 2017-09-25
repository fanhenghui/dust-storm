var socket = null;
var seriesUID = '';
var cellCanvases = null;
var cellSVGs = null;
var cells = null;
var protocRoot = null;

(function() {
    function getUserID(userName) {
        return userName + new Date().getTime() + Math.floor(Math.random() * 173 + 511);
    };

    function login(userName) {
        //add userName&userID attribute
        socket.userName = document.getElementById('username').innerHTML;
        socket.userID = getUserID(userName);

        socket = io.connect(SOCKET_IP);
        if(!socket) {
            //TODO log
            return;
        } else {
            socket.emit('login', {userid: socket.userID, username: socket.userName});
            socket.on('data' , function(tcpBuffer) {
                recvData(tcpBuffer, cmdHandler);
            });
        }
    };

    function logout() {
        if(socket != null) {
            socket.emit('disconnect', {userid: socket.userID, username: socket.userName});
            location.reload();
        }
    };

    function cmdHandler( cmdID, cellID, opID, tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader) {
      switch (cmdID) {
        case COMMAND_ID_BE_SEND_IMAGE:
            cells[i].handleJpegBuffer(tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader);
            break;
        case COMMAND_ID_BE_READY:
            // window.FE.triggerOnBE('test_uid');
            break;
        case COMMAND_ID_BE_SEND_WORKLIST:
            //showWorklist(tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader);
            break;
        case COMMAND_ID_BE_HEARTBEAT:
            socketClient.heartbeat();
            break;
        case COMMAND_ID_BE_SEND_ANNOTATION:
            // alert("delete selected annotation");
            //window.FE.changeAnnotation();
            break;
        default:
            break;
      }
    };

    function getProperCellSize() {
        var cellContainerW = document.getElementById('cell-container').offsetWidth;
        //var cellContainerH = document.getElementById('cell-container').offsetHeight;
        var navigatorHeight = document.getElementById('navigator-div').offsetHeight;
        var w = (cellContainerW - 20) / 2;
        var h = (window.innerHeight - navigatorHeight - 40) / 2;
        return {width: w, height: h};
    }

    function resize() {
        if(!socketClient.protocRoot) {
            //TODO LOG
            return;
        }
        var cellSize = getProperCellSize();
        var w = cellSize.width;
        var h = cellSize.height;
        for (var i = 0; i < 4; i++ ) {
            wbCells[i].resize(w,h);
        }

        var MsgResize = socketClient.protocRoot.lookup('medical_imaging.MsgResize');
        var msgResize = MsgResize.create();
        msgResize.cells.push({
          id: 0,
          type: 1,
          direction: 0,
          width: w,
          height: h
        });
        msgResize.cells.push({
          id: 1,
          type: 1,
          direction: 1,
          width: w,
          height: h
        });
        msgResize.cells.push({
          id: 2,
          type: 1,
          direction: 2,
          width: w,
          height: h
        });
        msgResize.cells.push({
          id: 3,
          type: 2,
          direction: 0,
          width: w,
          height: h
        });
        var msgBuffer = MsgResize.encode(msgResize).finish();
        socketClient.sendData(COMMAND_ID_FE_OPERATION, OPERATION_ID_RESIZE, 0, msgBuffer.byteLength, msgBuffer);
    }

    function prepare() {
        //Create cell object
        var cellContainer = document.getElementById('cell-container');
        cellCanvases = [
            document.getElementById('canvas0'), document.getElementById('canvas1'),
            document.getElementById('canvas2'), document.getElementById('canvas3')
        ];
        cellSVGs = [
            document.getElementById('svg0'), document.getElementById('svg1'),
            document.getElementById('svg2'), document.getElementById('svg3')
        ];

        //calcualte size

        for (var i = 0; i < 4; i++ ) {
            cells[i] = new Cell('cell_' + i, i, cellCanvases[i]. cellSVGs[i], socket);
            if(!cells[i].prepare()) {
                //TODO log
            }
        }

        //load protoc
        socketClient.loadProtoc(PROTOBUF_BE_FE);

        //register window quit linsener
        window.onbeforeunload = function(event) {
            logout();
        }        
    };

    prepare();
})();