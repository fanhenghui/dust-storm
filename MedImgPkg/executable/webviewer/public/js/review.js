(function() {
  // FE to BE
  COMMAND_ID_FE_SHUT_DOWN = 120000;
  COMMAND_ID_FE_READY = 120001;
  COMMAND_ID_FE_OPERATION = 120002;
  COMMAND_ID_FE_MPR_PLAY = 120003;
  COMMAND_ID_FE_VR_PLAY = 120004;
  COMMAND_ID_FE_SEARCH_WORKLIST = 120005;

  // BE to FE
  COMMAND_ID_BE_HEARTBEAT = 269999;
  COMMAND_ID_BE_READY = 270000;
  COMMAND_ID_BE_SEND_IMAGE = 270001;
  COMMAND_ID_BE_SEND_WORKLIST = 270002;
  COMMAND_ID_BE_SEND_ANNOTATION = 270003;

  // FE to BE Operation ID
  OPERATION_ID_INIT = 310000;
  OPERATION_ID_MPR_PAGING = 310001;
  OPERATION_ID_PAN = 310002;
  OPERATION_ID_ZOOM = 310003;
  OPERATION_ID_ROTATE = 310004;
  OPERATION_ID_WINDOWING = 310005;
  OPERATION_ID_RESIZE = 310006;
  OPERATION_ID_ANNOTATION = 310007;

  // init
  cellCanvas = [
    document.getElementById('canvas0'), document.getElementById('canvas1'),
    document.getElementById('canvas2'), document.getElementById('canvas3')
  ];
  cellJpeg = ['', '', '', ''];
  cellImage = [new Image(), new Image(), new Image(), new Image()];

  // init canvas size
  console.log('w:' + window.innerWidth);
  console.log('h:' + window.innerHeight);

  cellContainerWidth = document.getElementById('cell-container').offsetWidth;
  cellContainerHeight = document.getElementById('cell-container').offsetHeight;
  navigatorHeight = document.getElementById('navigator-div').offsetHeight;

  console.log('cells w:' + cellContainerWidth);
  console.log('cells h:' + cellContainerHeight);

  for (var index = 0; index < cellCanvas.length; index++) {
    cellCanvas[index].width = (cellContainerWidth - 20) / 2;
    cellCanvas[index].height = (window.innerHeight - navigatorHeight - 40) / 2;
  }

  function refreshCanvas(cellID) {
    cellCanvas[cellID].getContext('2d').drawImage(
        cellImage[cellID], 0, 0, cellCanvas[cellID].width,
        cellCanvas[cellID].height);
  }

  function handleImage(
      cellID, tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader) {
    //console.log('in render');
    if (withHeader) {  // receive a new image
      cellJpeg[cellID] = '';
    }

    var imgBuffer = new Uint8Array(tcpBuffer, bufferOffset, dataLen);
    cellJpeg[cellID] += String.fromCharCode.apply(null, imgBuffer);

    if (restDataLen <= 0) {
      cellImage[cellID].src = 'data:image/jpg;base64,' + btoa(cellJpeg[cellID]);
      cellImage[cellID].onload = function() {
        // console.log('Image Onload');
        refreshCanvas(cellID);
      };
    }
  }

  function msgHandle(
      cmdID, cellID, opID, tcpBuffer, bufferOffset, dataLen, restDataLen,
      withHeader) {
    switch (cmdID) {
      case COMMAND_ID_BE_SEND_IMAGE:
        handleImage(
            cellID, tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader);
        break;
      case COMMAND_ID_BE_READY:
        // window.FE.triggerOnBE('test_uid');
        break;
      case COMMAND_ID_BE_SEND_WORKLIST:
        showWorklist(tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader);
        break;
      case COMMAND_ID_BE_HEARTBEAT:
        window.FE.heartbeat();
        break;
      case COMMAND_ID_BE_SEND_ANNOTATION:
        // alert("delete selected annotation");
        window.FE.changeAnnotation();
        break;
      default:
        break;
    }
  }

  // TCP package related
  tcpPackageEnd =
      0;  // 0 msg header 1 data for last msg 2 msg header for last msg header
  msgCmdID = 0;
  msgCellID = 0;
  msgOpID = 0;
  msgRestDataLen = 0;
  lastMsgHeader = new ArrayBuffer(32);
  lastMsgHeaderLen = 0;

  function processTCPMsg(tcpBuffer) {
    var tcpPackageLen = tcpBuffer.byteLength;

    if (tcpPackageEnd == 0) {
      if (tcpPackageLen < 32) {
        tcpPackageEnd = 2;
        lastMsgHeaderLen = tcpPackageLen;
        tcpBuffer.copy(lastMsgHeader, 0, 0);
        return;
      }
      var header = new Uint32Array(tcpBuffer, 0, 8);
      msgCmdID = header[2];
      msgCellID = header[3];
      msgOpID = header[4];
      var lastMsgDatalen = header[7];

      if (tcpPackageLen - 32 == lastMsgDatalen) {  // completed one Msg
        msgHandle(
            msgCmdID, msgCellID, msgOpID, tcpBuffer, 32, lastMsgDatalen, 0,
            true);
      } else if (tcpPackageLen - 32 < lastMsgDatalen) {  // not completed one
        // Msg
        msgRestDataLen = lastMsgDatalen - (tcpPackageLen - 32);
        msgHandle(
            msgCmdID, msgCellID, msgOpID, tcpBuffer, 32, tcpPackageLen - 32,
            msgRestDataLen, true);
        tcpPackageEnd = 1;
      } else {  // this buffer carry next Msg
        // process current one
        msgHandle(
            msgCmdID, msgCellID, msgOpID, tcpBuffer, 32, lastMsgDatalen, 0,
            true);
        // recursion process rest
        var tcpBufferSub = tcpBuffer.slice(lastMsgDatalen + 32);
        tcpPackageEnd = 0;
        processTCPMsg(tcpBufferSub);
      }
    } else if (tcpPackageEnd == 1) {              // data for last msg
      if (tcpPackageLen - msgRestDataLen == 0) {  // complete last msg
        msgRestDataLen = 0;
        msgHandle(
            msgCmdID, msgCellID, msgOpID, tcpBuffer, 0, tcpPackageLen, 0,
            false);
      } else if (tcpPackageLen - msgRestDataLen < 0) {  // not complete data yet
        msgRestDataLen -= tcpPackageLen;
        tcpPackageEnd = 1;
        msgHandle(
            msgCmdID, msgCellID, msgOpID, tcpBuffer, 0, tcpPackageLen,
            msgRestDataLen, false);
      } else {  // this buffer carry next Msg
        msgHandle(
            msgCmdID, msgCellID, msgOpID, tcpBuffer, 0, msgRestDataLen, 0,
            false);
        var tcpBufferSub2 = tcpBuffer.slice(msgRestDataLen);
        msgRestDataLen = 0;
        tcpPackageEnd = 0;
        processTCPMsg(tcpBufferSub2);
      }
    } else if (tcpPackageEnd == 2) {  // msg header for last msg header
      var lastRestHeaderLen = 32 - lastMsgHeaderLen;
      if (tcpPackageLen <
          lastRestHeaderLen) {  // msg header is not completed yet
        tcpPackageEnd = 2;
        tcpBuffer.copy(lastMsgHeader, 0, lastRestHeaderLen, tcpPackageLen);
        lastMsgHeaderLen += tcpPackageLen;
        return;
      } else {  // msg header is completed
        tcpPackageEnd = 1;
        tcpBuffer.copy(lastMsgHeader, 0, lastRestHeaderLen, lastRestHeaderLen);
        var tcpBufferSub3 = tcpBuffer.slice(lastRestHeaderLen);

        var header2 = new Uint32Array(lastMsgHeader, 0, 8);
        msgCmdID = header2[2];
        msgCellID = header2[3];
        msgOpID = header2[4];
        msgRestDataLen = header2[7];

        tcpPackageEnd = 1;
        lastMsgHeaderLen = 0;
        processTCPMsg(tcpBufferSub3);
      }
    }
  }

  //////////////////////////////////////////////////////////////////
  // mouse event for test
  // ACTION ID
  ACTION_ID_ARROW = 0;
  ACTION_ID_ZOOM = OPERATION_ID_ZOOM;
  ACTION_ID_PAN = OPERATION_ID_PAN;
  ACTION_ID_ROTATE = OPERATION_ID_ROTATE;
  ACTION_ID_WINDOWING = OPERATION_ID_WINDOWING;


  BTN_NONE = -1;
  BTN_LEFT = 0;
  BTN_MIDDLE = 1;
  BTN_RIGHT = 2;

  BTN_DOWN = 0;
  BTN_UP = 1;

  // Btn status
  btnType = [BTN_NONE, BTN_NONE, BTN_NONE, BTN_NONE];
  btnStatus = [BTN_UP, BTN_UP, BTN_UP, BTN_UP];
  preMousePos = [{x: 0, y: 0}, {x: 0, y: 0}, {x: 0, y: 0}, {x: 0, y: 0}];
  // Current action
  curAction = ACTION_ID_ARROW;


  lastMouseMsg = {preX: 0, preY: 0, curX: 0, curY: 0};

  function mouseMoveEvent(event) {
    var cellname = event.toElement.id;
    if(!cellname) {return;};
    var cellid_s = cellname.slice(cellname.length - 1);
    var cellid = parseInt(cellid_s);
    if (btnStatus[cellid] != BTN_DOWN) {
      document.getElementById('test-info').innerText = '';
      return;
    }
    var button = event.button;
    var x = event.clientX - event.toElement.getBoundingClientRect().left;
    var y = event.clientY - event.toElement.getBoundingClientRect().top;
    document.getElementById('test-info').innerText = 'move cell id : ' +
        cellid.toString() + ' ' + x.toString() + ' ' + y.toString();

    if (preMousePos[cellid].x == x && preMousePos[cellid].y == y) {
      return;
    }
    if (button == BTN_LEFT) {
      // if (curAction == ACTION_ID_ARROW) {
      //  return;
      //}
      window.FE.mouseAction(
          cellid, {x: preMousePos[cellid].x, y: preMousePos[cellid].y},
          {x: x, y: y}, curAction);
    }

    preMousePos[cellid].x = x;
    preMousePos[cellid].y = y;
  }

  function mouseDownEvent(event) {
    var cellname = event.toElement.id;
    if(!cellname) {return;};
    var cellid_s = cellname.slice(cellname.length - 1);
    var cellid = parseInt(cellid_s);
    btnStatus[cellid] = BTN_DOWN;
    btnType[cellid] = event.button;

    var button = event.button;
    var x = event.clientX - event.toElement.getBoundingClientRect().left;
    var y = event.clientY - event.toElement.getBoundingClientRect().top;
    preMousePos[cellid].x = x;
    preMousePos[cellid].y = y;

    document.getElementById('test-info').innerText = 'down' + cellname;
  }

  function mouseUpEvent(event) {
    if(!cellname) {return;};
    var cellname = event.toElement.id;
    var cellid_s = cellname.slice(cellname.length - 1);
    var cellid = parseInt(cellid_s);
    btnStatus[cellid] = BTN_UP;
    btnType[cellid] = BTN_NONE;
    document.getElementById('test-info').innerText = 'up' + cellname;
  }

  // cellCanvas[0].addEventListener('mousemove', mouseMoveEvent);
  // cellCanvas[0].addEventListener('mousedown', mouseDownEvent);
  // cellCanvas[0].addEventListener('mouseup', mouseUpEvent);

  document.getElementById('svg0').addEventListener('mousemove', mouseMoveEvent);
  document.getElementById('svg0').addEventListener('mousedown', mouseDownEvent);
  document.getElementById('svg0').addEventListener('mouseup', mouseUpEvent);

  cellCanvas[1].addEventListener('mousemove', mouseMoveEvent);
  cellCanvas[1].addEventListener('mousedown', mouseDownEvent);
  cellCanvas[1].addEventListener('mouseup', mouseUpEvent);

  cellCanvas[2].addEventListener('mousemove', mouseMoveEvent);
  cellCanvas[2].addEventListener('mousedown', mouseDownEvent);
  cellCanvas[2].addEventListener('mouseup', mouseUpEvent);

  cellCanvas[3].addEventListener('mousemove', mouseMoveEvent);
  cellCanvas[3].addEventListener('mousedown', mouseDownEvent);
  cellCanvas[3].addEventListener('mouseup', mouseUpEvent);


  searchBtn = document.getElementById('searchBtn');
  loadBtn = document.getElementById('loadBtn');

  var worklistBuffer;
  function showWorklist(
      tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader) {
    console.log('prepare data 4 worklist!');
    if (withHeader) {
      worklistBuffer = new ArrayBuffer(dataLen);
    }

    // TODO: handle multiple parts later, now assume passing as a whole
    var dstview = new Uint8Array(worklistBuffer);
    var srcview = new Uint8Array(tcpBuffer, bufferOffset, dataLen);

    for (var i = 0; i < dataLen; i++) {
      dstview[i] = srcview[i];
    }

    if (restDataLen <= 0) {
      window.FE.showWorklist();
    }
  };

  function searchWorkList(event) {
    console.log('searchWorkList');
    window.FE.searchWorkList();
  }

  // function loadOneSeries(event)
  // {
  //   console.log('loadOneSeries');
  //   // get the element currently selected by usr
  //   wl = document.getElementById('worklist');
  //   // TODO: get seriesID from $("#worklist")

  //   // load this series
  //   window.FE.triggerOnBE('test_uid');
  // }

  searchBtn.addEventListener('click', searchWorkList);
  // loadBtn.addEventListener('click', loadOneSeries);

  $("#loadBtn")
      .on('click', function(e) {
        var series_id = $("#table tbody tr.success td:nth-child(3)").html();
        console.log('loadOneSeries: ', series_id);
        alert(series_id);
        document.getElementById("worklist-div").hidden = true;
        document.getElementById("review-div").hidden = false;
        window.FE.triggerOnBE(series_id);  //'test_uid'
      });


  var selectedElement = 0;
  var moveType = 0;
  var currentX = 0;
  var currentY = 0;

  function moveElement(evt) {
    dx = evt.clientX - currentX;
    dy = evt.clientY - currentY;

    if (selectedElement != 0) {
      if (moveType == BTN_LEFT) {
        //   currentMatrix[4] += dx;
        //   currentMatrix[5] += dy;
        //   newMatrix = "matrix(" + currentMatrix.join(' ') + ")";
        //   selectedElement.setAttributeNS(null, "transform", newMatrix);

        var cx = parseFloat(selectedElement.getAttribute('cx'));
        selectedElement.setAttribute('cx', cx + dx);

        var cy = parseFloat(selectedElement.getAttribute('cy'));
        selectedElement.setAttribute('cy', cy + dy);
      } else if (moveType == BTN_RIGHT) {
        var cx = selectedElement.getAttribute('cx');
        var cy = selectedElement.getAttribute('cy');
        
        // TODO: capture the right canvas
        var bBox = document.getElementById("canvas0").getBoundingClientRect();
        // console.log(bBox.left);
        // console.log(bBox.top);
        var r = Math.sqrt(
            (evt.clientX - bBox.left - cx) * (evt.clientX - bBox.left - cx) +
            (evt.clientY - bBox.top - cy) * (evt.clientY - bBox.top - cy));
        selectedElement.setAttribute('r', r);
        }
    }
    currentX = evt.clientX;
    currentY = evt.clientY;
  }

  function deselectElement(evt) {
    if (selectedElement != 0) {

      // Tell BE my new status
      // canvas_0, primitive type (0: circle), data
      window.FE.sendAnnotation(0, 0, 
        {
          cx: parseFloat(selectedElement.getAttribute('cx')),
          cy: parseFloat(selectedElement.getAttribute('cy')),
          r: parseFloat(selectedElement.getAttribute('r'))
        }); 

      // selectedElement.removeAttributeNS(null, "onmousemove");
      document.onmousemove = null;
      selectedElement.removeAttributeNS(null, "onmouseout");
      selectedElement.removeAttributeNS(null, "onmouseup");
      selectedElement = 0;
      moveType = 0;
    }
  }

  window.FE = {
    username: null,
    userid: null,
    socket: null,
    mouseActionTik: new Date().getTime(),
    series_uid: "",


    genUID: function(username) {
      return username + new Date().getTime() +
          Math.floor(Math.random() * 173 + 511);
    },

    init: function(username) {
      //客户端根据时间和随机数生成uid,以用户名称可以重复，后续改为数据库的UID名称
      this.userid = this.genUID(username);
      this.username = username;

      //链接websocket服务器
      this.socket = io.connect('http://127.0.0.1:8000');

      //通知服务器有用户登录 TODO 这段逻辑应该在登录的时候做
      this.socket.emit('login', {userid: this.userid, username: this.username});

      this.socket.on('data', function(arraybuffer) {
        //console.log('receive data.');
        processTCPMsg(arraybuffer);
      });

    },

    heartbeat: function() {
      this.socket.emit(
          'heartbeat', {userid: this.userid, username: this.username});
    },

    userLogOut: function() {
      this.socket.emit(
          'disconnect', {userid: this.userid, username: this.username});
      location.reload();
    },

    userLogIn: function() {
      var username = document.getElementById('username').innerHTML;
      this.init(username);
    },

    switchCommonTool: function(btnID) {
      switch (btnID) {
        case 'common-tool-arrow':
          curAction = ACTION_ID_ARROW;
          document.getElementById('test-info').innerText = 'action arrow';
          break;
        case 'common-tool-zoom':
          curAction = ACTION_ID_ZOOM;
          document.getElementById('test-info').innerText = 'action zoom';
          break;
        case 'common-tool-pan':
          curAction = ACTION_ID_PAN;
          document.getElementById('test-info').innerText = 'action pan';
          break;
        case 'common-tool-rotate':
          curAction = ACTION_ID_ROTATE;
          document.getElementById('test-info').innerText = 'action rotate';
          break;
        case 'common-tool-windowing':
          curAction = ACTION_ID_WINDOWING;
          document.getElementById('test-info').innerText = 'action windowing';
          break;
        case 'common-tool-annotation':
          // TODO annotation
          // curAction = ACTION_ID_ARROW;
          break;
        default:
          // TODO ERR
          break;
      }
    },

    mouseAction: function(cellid, prePos, curPos, actionID) {
      var binding_func =
          (function(err, root) {
            if (err) {
              console.log('load proto failed!');
              throw err;
            }

            var curTick = new Date().getTime();
            if (Math.abs(window.FE.mouseActionTik - curTick) < 10) {
              return;
            }
            window.FE.mouseActionTik = curTick;
            if (cellid == 0 || cellid == 1 || cellid == 2) {
              if (actionID == ACTION_ID_ARROW) {
                actionID = OPERATION_ID_MPR_PAGING;
              } else if (actionID == ACTION_ID_ROTATE) {
                return;
              }
            } else {
              if (actionID == ACTION_ID_ARROW) {
                actionID = ACTION_ID_ROTATE;
              }
            }

            var MsgMouse = root.lookup('medical_imaging.MsgMouse');
            var msgMouse = MsgMouse.create({
              pre: {x: prePos.x, y: prePos.y},
              cur: {x: curPos.x, y: curPos.y},
              tag: 0
            });
            var msgBuffer = MsgMouse.encode(msgMouse).finish();
            var msgLength = msgBuffer.byteLength;
            var cmdBuffer = new ArrayBuffer(32 + msgLength);
            // header
            var header = new Uint32Array(cmdBuffer, 0, 8);
            header[0] = 0;
            header[1] = 0;
            header[2] = COMMAND_ID_FE_OPERATION;
            header[3] = cellid;
            header[4] = actionID;
            header[5] = 0;
            header[6] = 0;
            header[7] = msgLength;
            // data
            var srcBuffer = new Uint8Array(msgBuffer);
            var dstBuffer = new Uint8Array(cmdBuffer, 8 * 4, msgLength);
            for (var index = 0; index < msgLength; index++) {
              dstBuffer[index] = srcBuffer[index];
            }
            console.log('emit mouse action message.');

            this.socket.emit('data', {
              userid: this.userid,
              username: this.username,
              content: cmdBuffer
            });
          }).bind(this);

      protobuf.load('./data/mi_message.proto', binding_func);
    },

    triggerOnBE: function(series_uid) {
      this.series_uid = series_uid;
      var binding_func =
          (function(err, root) {
            if (err) {
              console.log('load proto failed!');
              throw err;
            }
            var MsgInit = root.lookup('medical_imaging.MsgInit');
            var msgInit = MsgInit.create();
            msgInit.seriesUid = this.series_uid;
            msgInit.pid = 1000;

            // adjust width&height
            cellContainerWidth =
                document.getElementById('cell-container').offsetWidth;
            cellContainerHeight =
                document.getElementById('cell-container').offsetHeight;
            navigatorHeight =
                document.getElementById('navigator-div').offsetHeight;

            console.log('cells w:' + cellContainerWidth);
            console.log('cells h:' + cellContainerHeight);

            for (var index = 0; index < cellCanvas.length; index++) {
              cellCanvas[index].width = (cellContainerWidth - 20) / 2;
              cellCanvas[index].height =
                  (window.innerHeight - navigatorHeight - 40) / 2;
            }


            // MPR
            msgInit.cells.push({
              id: 0,
              type: 1,
              direction: 0,
              width: cellCanvas[0].width,
              height: cellCanvas[0].height
            });
            
            d3.select("#svg0")
            .attr('width', cellCanvas[0].width)
            .attr('height', cellCanvas[0].height);

            msgInit.cells.push({
              id: 1,
              type: 1,
              direction: 1,
              width: cellCanvas[1].width,
              height: cellCanvas[1].height
            });
            msgInit.cells.push({
              id: 2,
              type: 1,
              direction: 2,
              width: cellCanvas[2].width,
              height: cellCanvas[2].height
            });
            msgInit.cells.push({
              id: 3,
              type: 2,
              direction: 0,
              width: cellCanvas[3].width,
              height: cellCanvas[3].height
            });

            var msgBuffer = MsgInit.encode(msgInit).finish();
            var msgLength = msgBuffer.byteLength;
            var cmdBuffer = new ArrayBuffer(32 + msgLength);

            // header
            var header = new Uint32Array(cmdBuffer, 0, 8);
            header[0] = 0;
            header[1] = 0;
            header[2] = COMMAND_ID_FE_OPERATION;
            header[3] = 0;
            header[4] = OPERATION_ID_INIT;
            header[5] = 0;
            header[6] = 0;
            header[7] = msgLength;

            // data
            var srcBuffer = new Uint8Array(msgBuffer);
            var dstBuffer = new Uint8Array(cmdBuffer, 8 * 4, msgLength);
            for (var index = 0; index < msgLength; index++) {
              dstBuffer[index] = srcBuffer[index];
            }
            console.log('emit trigger BE message.');

            this.socket.emit('data', {
              userid: this.userid,
              username: this.username,
              content: cmdBuffer
            });
          }).bind(this);

      protobuf.load('./data/mi_message.proto', binding_func);
    },

    resize: function() {
      console.log('resize');
      var binding_func =
          (function(err, root) {
            if (err) {
              console.log('load proto failed!');
              throw err;
            }
            cellContainerWidth =
                document.getElementById('cell-container').offsetWidth;
            cellContainerHeight =
                document.getElementById('cell-container').offsetHeight;
            navigatorHeight =
                document.getElementById('navigator-div').offsetHeight;

            console.log('cells w:' + cellContainerWidth);
            console.log('cells h:' + cellContainerHeight);

            for (var index = 0; index < cellCanvas.length; index++) {
              cellCanvas[index].width = (cellContainerWidth - 20) / 2;
              cellCanvas[index].height =
                  (window.innerHeight - navigatorHeight - 40) / 2;
            }

            document.getElementById("svg0").width = cellCanvas[0].width;
            document.getElementById("svg0").height = cellCanvas[0].height;

            var MsgResize = root.lookup('medical_imaging.MsgResize');
            var msgResize = MsgResize.create();

            // MPR
            msgResize.cells.push({
              id: 0,
              type: 1,
              direction: 0,
              width: cellCanvas[0].width,
              height: cellCanvas[0].height
            });
            msgResize.cells.push({
              id: 1,
              type: 1,
              direction: 1,
              width: cellCanvas[1].width,
              height: cellCanvas[1].height
            });
            msgResize.cells.push({
              id: 2,
              type: 1,
              direction: 2,
              width: cellCanvas[2].width,
              height: cellCanvas[2].height
            });
            msgResize.cells.push({
              id: 3,
              type: 2,
              direction: 0,
              width: cellCanvas[3].width,
              height: cellCanvas[3].height
            });
            var msgBuffer = MsgResize.encode(msgResize).finish();
            var msgLength = msgBuffer.byteLength;
            var cmdBuffer = new ArrayBuffer(32 + msgLength);
            // header
            var header = new Uint32Array(cmdBuffer, 0, 8);
            header[0] = 0;
            header[1] = 0;
            header[2] = COMMAND_ID_FE_OPERATION;
            header[3] = 0;
            header[4] = OPERATION_ID_RESIZE;
            header[5] = 0;
            header[6] = 0;
            header[7] = msgLength;
            // data
            var srcBuffer = new Uint8Array(msgBuffer);
            var dstBuffer = new Uint8Array(cmdBuffer, 8 * 4, msgLength);
            for (var index = 0; index < msgLength; index++) {
              dstBuffer[index] = srcBuffer[index];
            }
            console.log('emit resize message.');

            this.socket.emit('data', {
              userid: this.userid,
              username: this.username,
              content: cmdBuffer
            });
          }).bind(this);

      protobuf.load('./data/mi_message.proto', binding_func);
    },

    loadSeries: function() {
      var header_buffer = new ArrayBuffer(32);
      var header = new Uint32Array(header_buffer);
      header[0] = 0;
      header[1] = 0;
      header[2] = COMMAND_ID_FE_LOAD_SERIES;
      header[3] = 0;  // paging
      header[4] = 0;
      header[5] = 0;
      header[6] = 0;
      header[7] = 0;
      this.socket.emit('data', {
        userid: this.userid,
        username: this.username,
        content: header_buffer
      })
    },

    searchWorkList: function() {
      var header_buffer = new ArrayBuffer(32);
      var header = new Uint32Array(header_buffer);
      header[0] = 0;
      header[1] = 0;
      header[2] = COMMAND_ID_FE_SEARCH_WORKLIST;
      header[3] = 0;
      header[4] = 0;
      header[5] = 0;
      header[6] = 0;
      header[7] = 0;

      this.socket.emit('data', {
        userid: this.userid,
        username: this.username,
        content: header_buffer
      });
      // for test
      // this.socket.emit('message', {
      //   userid: this.userid,
      //   username: this.username,
      //   content: 'searchWorkList'
      // });
    },

    showWorklist: function() {
      protobuf.load('./data/mi_message.proto', function(err, root) {
        if (err) {
          console.log('load proto failed!');
          throw err;
        }
        var MsgWorklistType = root.lookup('medical_imaging.MsgWorklist');
        var worklistView = new Uint8Array(worklistBuffer);
        var message = MsgWorklistType.decode(worklistView);
        console.log(worklist);
        var obj = MsgWorklistType
                      .toObject(message, {
                        patient_name: String,
                        patient_id: String,
                        series_uid: String,
                        imaging_modality: String
                      })
                      .items;

        var tbody = document.getElementById("worklist");
        tbody.innerHTML = "";
        for (var i = 0; i < obj.length; i++) {
          var tr = "<tr>";
          for (var propt in obj[i]) {
            tr += "<td>" + obj[i][propt] + "</td>"
          }
          tr += "</tr>";

          /* We add the table row to the table body */
          tbody.innerHTML += tr;
        }

        $("#table tbody tr")
            .click(function() {
              $(this).addClass('success').siblings().removeClass('success');
              // var value = $(this).find('td:nth-child(3)').html();
              // alert(value);
            });
      });
    },

    playVR: function() {
      var header_buffer = new ArrayBuffer(32);
      var header = new Uint32Array(header_buffer);
      header[0] = 0;
      header[1] = 0;
      header[2] = COMMAND_ID_FE_VR_PLAY;
      header[3] = 3;  // VR cell id
      header[4] = 0;
      header[5] = 0;
      header[6] = 0;
      header[7] = 0;
      this.socket.emit('data', {
        userid: this.userid,
        username: this.username,
        content: header_buffer
      });
    },

    addAnnotation: function() {
      // add a circle
      var newCircle = d3.select("#svg0").append("circle");
      newCircle.attr("cx", 19)
          .attr("cy", 21)
          .attr("r", 10)
          .style("fill-opacity", 0.0)//热点是整个圆
          //.style("fill", 'none')//热点是圆圈
          .style("stroke", "red")
          .style("stroke-opacity", 0.8)
          .style("stroke-width", 2)
          .style("cursor", "move")
          .on("contextmenu", function(data, index) {
            d3.event.preventDefault();
            return false;
          });

      newCircle.on("mousedown", function() {
        selectedElement = event.target;
        if (!selectedElement) {
          console.log("select no element");
          return ;
        }
        if (event.button == BTN_LEFT)  // move circle
        {
          currentX = event.clientX;
          currentY = event.clientY;
          moveType = BTN_LEFT;
        }

        else if (event.button == BTN_RIGHT)  // change circle radius
        {
          moveType = BTN_RIGHT;
        }

        // selectedElement.setAttributeNS(null, "onmousemove",
        // "moveElement(evt)");
        document.onmousemove = moveElement;

        // selectedElement.setAttributeNS(null, "onmouseout",
        // "deselectElement(evt)");

        document.onmouseup = deselectElement;
        // selectedElement.setAttributeNS(null, "onmouseup",
        // "deselectElement(evt)");
        // document.getElementById("svg0").appendChild(selectedElement);
      });

      // emit msg to server
      // canvas_0, primitive type (0: circle), data
      this.sendAnnotation(0, 0, 
        {
          cx: parseFloat(newCircle.attr('cx')),
          cy: parseFloat(newCircle.attr('cy')),
          r: parseFloat(newCircle.attr('r'))
        }); 
    },

    removeAnnotation : function ()
    {
      var cmdBuffer = new ArrayBuffer(32);
      var header = new Uint32Array(cmdBuffer, 0, 8);
      header[0] = 0;
      header[1] = 0;
      header[2] = COMMAND_ID_FE_OPERATION;
      header[3] = 0; // cellid;
      header[4] = OPERATION_ID_ANNOTATION;
      header[5] = 0;
      header[6] = 0;
      header[7] = 0;

      this.socket.emit('data', {
        userid: this.userid,
        username: this.username,
        content: cmdBuffer
      });
    },

    sendAnnotation : function(cellid, annotationType, annotationData)
    {
      var annotationProto = (function(err, root){
        if (err) {
          console.log('load proto failed!');
          throw err;
        }

        var MsgAnnotation = root.lookup('medical_imaging.MsgAnnotation');
        var annotation = MsgAnnotation.create({
          type: annotationType,
          para1: annotationData.cx,
          para2: annotationData.cy,
          para3: annotationData.r,
        });

        var msgBuffer = MsgAnnotation.encode(annotation).finish();
        var msgLength = msgBuffer.byteLength;
        var cmdBuffer = new ArrayBuffer(32 + msgLength);
        // header
        var header = new Uint32Array(cmdBuffer, 0, 8);
        header[0] = 0;
        header[1] = 0;
        header[2] = COMMAND_ID_FE_OPERATION;
        header[3] = cellid;
        header[4] = OPERATION_ID_ANNOTATION;
        header[5] = 0;
        header[6] = 0;
        header[7] = msgLength;
        // data
        var srcBuffer = new Uint8Array(msgBuffer);
        var dstBuffer = new Uint8Array(cmdBuffer, 8 * 4, msgLength);
        for (var index = 0; index < msgLength; index++) {
          dstBuffer[index] = srcBuffer[index];
        }
        console.log('emit annotation location message.');

        this.socket.emit('data', {
          userid: this.userid,
          username: this.username,
          content: cmdBuffer
        });
      }).bind(this);
      protobuf.load('./data/mi_message.proto', annotationProto);
    },

    changeAnnotation : function ()
    {
      d3.selectAll("circle").remove();
      // d3.select("#" + myDatum.data.uniqueID);
    }
    // changeLayout1x1: function() {
    //     document.getElementById("cell1").style.visibility = "hidden";
    //     document.getElementById("cell2").style.visibility = "hidden";
    //     document.getElementById("cell3").style.visibility = "hidden";

    //     var cell0 = document.getElementById("cell0").style.visibility =
    //     "visible";
    //     cell0.width = 1040;
    //     cell0.height = 1024;

    //     document.getElementById("cell0Canvas").width = 1024;
    //     document.getElementById("cell0Canvas").height = 1024;
    // },

    // changeLayout2x2: function() {
    //     document.getElementById("cell1").style.visibility = "visible";
    //     document.getElementById("cell2").style.visibility = "visible";
    //     document.getElementById("cell3").style.visibility = "visible";

    //     var cell0 = document.getElementById("cell0").style.visibility =
    //     "visible";
    //     cell0.width = 518;
    //     cell0.height = 518;

    //     document.getElementById("cell0Canvas").width = 512;
    //     document.getElementById("cell0Canvas").height = 512;
    // }

  },

  window.FE.userLogIn();
  window.onbeforeunload = function(event) {
    window.FE.userLogOut();
    return message;
  };

  var comToolsDiv = document.getElementById('common-tools');
  var comToolsBtns = comToolsDiv.getElementsByTagName('button');
  for (var i = 0; i < comToolsBtns.length; ++i) {
    comToolsBtns[i].addEventListener('click', function(event) {
      FE.switchCommonTool(this.id);
    });
  }

  var btnVRRotate = document.getElementById('btn-play-vr');
  btnVRRotate.addEventListener('click', function() {
    FE.playVR();
  });

  var btnAddAnnotation = document.getElementById('btn-addannotation');
  btnAddAnnotation.addEventListener('click', function() {
    FE.addAnnotation();
  });

  var btnRemoveAnnotation = document.getElementById('btn-removeannotation');
  btnRemoveAnnotation.addEventListener('click', function() {
    FE.removeAnnotation();
  });

  window.onresize = window.onresize = function() { window.FE.resize(); };

})()