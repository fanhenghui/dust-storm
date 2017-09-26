var socket = null;
var seriesUID = '';
var cellCanvases = null;
var cellSVGs = null;
var cells = null;
var protocRoot = null;
var worklistBuffer = null;
var socketClient = null;
var revcBEReady = false;

(function() {
  function getUserID(userName) {
    return userName + new Date().getTime() +
        Math.floor(Math.random() * 173 + 511);
  }

  function login() {
    socket = io.connect(SOCKET_IP);
    if (!socket) {
      // TODO log
      return;
    } else {
      // add userName&userID attribute
      socket.userName = document.getElementById('username').innerHTML;
      socket.userID = getUserID(socket.userName);
      //create socketClient
      socketClient = new SocketClient(socket);
      socketClient.loadProtoc(PROTOBUF_BE_FE);
      socket.emit('login', {userid: socket.userID, username: socket.userName});
      socket.on('data', function(tcpBuffer) { socketClient.recvData(tcpBuffer, cmdHandler); });
    }
  }

  function logout() {
    if (socket != null) {
      socket.emit(
          'disconnect', {userid: socket.userID, username: socket.userName});
      location.reload();
    }
  }

  function recvWorklist(tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader) {
    if (!socketClient.protocRoot) {
      // TODO LOG
      return;
    }
    if (withHeader) {
      worklistBuffer = new ArrayBuffer(dataLen + restDataLen);
    }

    var dstview = new Uint8Array(worklistBuffer);
    var srcview = new Uint8Array(tcpBuffer, bufferOffset, dataLen);
    var cpSrc = worklistBuffer.byteLength - (dataLen + restDataLen);
    for (var i = cpSrc; i < dataLen; i++) {
      dstview[i] = srcview[i];
    }

    if (restDataLen <= 0) {
      console.log('recv worklist.');
      var MsgWorklistType = socketClient.protocRoot.lookup('medical_imaging.MsgWorklist');
      var worklistView = new Uint8Array(worklistBuffer);
      var message = MsgWorklistType.decode(worklistView);
      console.log(worklist);
      var obj = MsgWorklistType.toObject(message, {
                      patient_name: String,
                      patient_id: String,
                      series_uid: String,
                      imaging_modality: String
                    }).items;
      var tbody = document.getElementById('worklist');
      tbody.innerHTML = '';
      for (var i = 0; i < obj.length; i++) {
        var tr = '<tr>';
        for (var propt in obj[i]) {
          tr += '<td>' + obj[i][propt] + '</td>';
        }
        tr += '</tr>';
        tbody.innerHTML += tr;
      }
    }

    //style changed when choose tr (based on bootstrap)
    $('#table tbody tr').click(function() {
        $(this).addClass('success').siblings().removeClass('success');
    });
  }

  function cmdHandler(cmdID, cellID, opID, tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader) {
    switch (cmdID) {
      case COMMAND_ID_BE_SEND_IMAGE:
        cells[cellID].handleJpegBuffer(tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader);
        break;
      case COMMAND_ID_BE_READY:
        revcBEReady = true;
        break;
      case COMMAND_ID_BE_SEND_WORKLIST:
        recvWorklist(tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader);
        break;
      case COMMAND_ID_BE_HEARTBEAT:
        socketClient.heartbeat();
        break;
      case COMMAND_ID_BE_SEND_ANNOTATION: 
        break;
      default:
        break;
    }
  }

  function getProperCellSize() {
    var cellContainerW = document.getElementById('cell-container').offsetWidth;
    // var cellContainerH = document.getElementById('cell-container').offsetHeight;
    var navigatorHeight = document.getElementById('navigator-div').offsetHeight;
    var w = (cellContainerW - 20) / 2;
    var h = (window.innerHeight - navigatorHeight - 40) / 2;
    return {width: w, height: h};
  }

  function resize() {
    if (!revcBEReady) {
      // TODO LOG
      return;
    }
    if (seriesUID === '') {
      // TODO LOG
      return;
    }
    if (!socketClient.protocRoot) {
      // TODO LOG
      return;
    }
    var cellSize = getProperCellSize();
    var w = cellSize.width;
    var h = cellSize.height;
    for (var i = 0; i < cells.length; i++) {
      cells[i].resize(w, h);
    }

    var MsgResize = socketClient.protocRoot.lookup('medical_imaging.MsgResize');
    var msgResize = MsgResize.create();
    msgResize.cells.push({id: 0, type: 1, direction: 0, width: w, height: h});
    msgResize.cells.push({id: 1, type: 1, direction: 1, width: w, height: h});
    msgResize.cells.push({id: 2, type: 1, direction: 2, width: w, height: h});
    msgResize.cells.push({id: 3, type: 2, direction: 0, width: w, height: h});
    var msgBuffer = MsgResize.encode(msgResize).finish();
    socketClient.sendData(COMMAND_ID_FE_OPERATION, OPERATION_ID_RESIZE, 0, msgBuffer.byteLength, msgBuffer);
  }

  function loadSeries(series) {
    seriesUID = series;
    if (!socketClient) {
      // TODO LOG
      return;
    }

    //create cells
    var cellSize = getProperCellSize();
    var w = cellSize.width;
    var h = cellSize.height;
    cells = [null, null, null, null];
    for (var i = 0; i < 4; i++) {
      var cellName = 'cell_' +i;
      var canvas = cellCanvases[i];
      var svg = cellSVGs[i];
      cells[i] = new Cell(cellName, i, canvas, svg, socketClient);
      cells[i].resize(cellSize.width, cellSize.height);
      if (!cells[i].prepare()) {
        // TODO log
      }
    }

    //init default cell action
    cells[0].mouseAction = ACTION_ID_MPR_PAGING;
    cells[1].mouseAction = ACTION_ID_MPR_PAGING;
    cells[2].mouseAction = ACTION_ID_MPR_PAGING;
    cells[3].mouseAction = ACTION_ID_ROTATE;

    //nofity BE
    var MsgInit = socketClient.protocRoot.lookup('medical_imaging.MsgInit');
    var msgInit = MsgInit.create();
    msgInit.seriesUid = seriesUID;
    msgInit.pid = 1000;
    msgInit.cells.push({id: 0, type: 1, direction: 0, width: w, height: h});
    msgInit.cells.push({id: 1, type: 1, direction: 1, width: w, height: h});
    msgInit.cells.push({id: 2, type: 1, direction: 2, width: w, height: h});
    msgInit.cells.push({id: 3, type: 2, direction: 0, width: w, height: h});

    var msgBuffer = MsgInit.encode(msgInit).finish();
    socketClient.sendData(COMMAND_ID_FE_OPERATION, OPERATION_ID_INIT, 0, msgBuffer.byteLength,msgBuffer);
  }

  function switchCommonTool(btnID) {
    document.getElementById('test-info').innerText = btnID;
    switch (btnID) {
      case 'common-tool-arrow':
        cells[0].mouseAction = ACTION_ID_MPR_PAGING;
        cells[1].mouseAction = ACTION_ID_MPR_PAGING;
        cells[2].mouseAction = ACTION_ID_MPR_PAGING;
        cells[3].mouseAction = ACTION_ID_ROTATE;
        break;
      case 'common-tool-zoom':
        for (var i = 0; i < cells.length; ++i) {
          cells[i].mouseAction = ACTION_ID_ZOOM;
        }
        break;
      case 'common-tool-pan':
        for (var i = 0; i < cells.length; ++i) {
          cells[i].mouseAction = ACTION_ID_PAN;
        }
        break;
      case 'common-tool-rotate':
        for (var i = 0; i < cells.length; ++i) {
          cells[i].mouseAction = ACTION_ID_ROTATE;
        }
        break;
      case 'common-tool-windowing':
        for (var i = 0; i < cells.length; ++i) {
          cells[i].mouseAction = ACTION_ID_WINDOWING;
        }
        break;
      case 'common-tool-annotation':
        // TODO annotation
        break;
      default:
        // TODO ERR
        break;
    }
  }

  function searchWorklist() {
    if (!revcBEReady) {
      // TODO LOG
      console.log('BE not ready!');
      alert('BE not ready!');
      return;
    }
    socketClient.sendData(COMMAND_ID_FE_SEARCH_WORKLIST,0,0,null);
  }

  function playVR() {
    socketClient.sendData(COMMAND_ID_FE_VR_PLAY,0,3,null);
  }

  function prepare() {
    // Create cell object
    var cellContainer = document.getElementById('cell-container');
    cellCanvases = [
      document.getElementById('canvas0'), document.getElementById('canvas1'),
      document.getElementById('canvas2'), document.getElementById('canvas3')
    ];
    cellSVGs = [
      document.getElementById('svg0'), document.getElementById('svg1'),
      document.getElementById('svg2'), document.getElementById('svg3')
    ];  

    // register button event
    var searchWorklistBtn = document.getElementById('searchBtn');
    searchWorklistBtn.addEventListener('click', function(event) { searchWorklist(); });

    var loadSeriesBtn = document.getElementById('loadBtn');
    loadSeriesBtn.addEventListener('click', function(event) {
      var series = $('#table tbody tr.success td:nth-child(3)').html();
      if (!series) {
        alert('please choose one series.');
        reutrn;
      }
      document.getElementById('worklist-div').hidden = true;
      document.getElementById('review-div').hidden = false;
      loadSeries(series);
    });

    var comToolsDiv = document.getElementById('common-tools');
    var comToolsBtns = comToolsDiv.getElementsByTagName('button');
    for (var i = 0; i < comToolsBtns.length; ++i) {
      comToolsBtns[i].addEventListener('click', function(event) {
        switchCommonTool(this.id);
      });
    }
    
    var playVRBtn = document.getElementById('btn-play-vr');
    playVRBtn.addEventListener('click', function() {
      playVR();
    });

    // register window quit linsener
    window.onbeforeunload = function(event) { logout(); }
    window.onresize = function() {resize()};
    var username = document.getElementById('username').innerHTML;
    login();
  }

  prepare();
})();