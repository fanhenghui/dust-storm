var socket = null;
var seriesUID = '';
var cellCanvases = null;
var cellSVGs = null;
var cells = null;
var protocRoot = null;
var worklistBuffer = null;
var socketClient = null;
var revcBEReady = false;

var annotationListBuffer = null;
var annotationTable = null;

//layout parameter
const LAYOUT_1X1 = '1x1';
const LAYOUT_2X2 = '2x2';
var layoutStatus = LAYOUT_2X2;
var maxCellID = -1;

(function() {
    function getUserID(userName) {
        return userName + new Date().getTime() + Math.floor(Math.random() * 173 + 511);
    }

    function login() {
        socket = io.connect(SOCKET_IP);
        if (!socket) {
            console.log('connect server failed.');
            alert('connect server failed.');
            return;
        } else {
            // add userName&userID attribute
            socket.userName = document.getElementById('username').innerHTML;
            socket.userID = getUserID(socket.userName);
            //create socketClient
            socketClient = new SocketClient(socket);
            socketClient.loadProtoc(PROTOBUF_BE_FE);
            socket.emit('login', {
                userid: socket.userID,
                username: socket.userName
            });
            socket.on('data', function(tcpBuffer) {
                socketClient.recvData(tcpBuffer, cmdHandler);
            });
        }
    }

    function logout() {
        if (socket != null) {
            socket.emit('disconnect', {userid: socket.userID,username: socket.userName});
            location.reload();
        }
    }

    function recvWorklist(tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader) {
        if (!socketClient.protocRoot) {
            console.log('null protocbuf.');
            return;
        }
        if (withHeader) {
            worklistBuffer = new ArrayBuffer(dataLen + restDataLen);
        }

        var dstview = new Uint8Array(worklistBuffer);
        var srcview = new Uint8Array(tcpBuffer, bufferOffset, dataLen);
        var cpSrc = worklistBuffer.byteLength - (dataLen + restDataLen);
        for (var i = 0; i < dataLen; i++) {
            dstview[i+cpSrc] = srcview[i];
        }

        if (restDataLen <= 0) {
            console.log('recv worklist.');
            var MsgWorklistType = socketClient.protocRoot.lookup('medical_imaging.MsgWorklist');
            if (!MsgWorklistType) {
                console.log('get worklist message type failed.');
            }
            var worklistView = new Uint8Array(worklistBuffer);
            var message = MsgWorklistType.decode(worklistView);
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

    function annoListDeleteRow(row) {
        var annoTableRows = annotationTable.rows;
        if (annoTableRows.length > row + 1) {
            annotationTable.deleteRow(row + 1);
        }
    }

    function annoListAddRow(row,id,cx,cy,cz,diameter,info) {
        if (document.getElementById(id)) {
            console.log('add repeated row');
            return;
        }
        var rowItem = annotationTable.insertRow(row + 1);
        $(rowItem).click(function(event) {
            $(this).addClass('success').siblings().removeClass('success');
            //send focus annotation message
            var anno_id = $(this).attr('id');
            if (anno_id) {
                sendAnnotationMSG(0, 0, anno_id, ANNOTATION_FOCUS, true, 0, 0, 0, socketClient); 
            }
        });
        rowItem.setAttribute('id',id);
        rowItem.insertCell(0).innerHTML = cx.toFixed(2) + ',' + cy.toFixed(2) + ',' + cz.toFixed(2);
        rowItem.insertCell(1).innerHTML = diameter.toFixed(2);
        rowItem.insertCell(2).innerHTML = info;
    }

    function annoListModifyRow(row,cx,cy,cz,diameter,info) {
        var annoTableRows = annotationTable.rows;
        if (annoTableRows.length > row + 1) {
            var annoCell = annoTableRows[row + 1].cells;
            annoCell[0].innerHTML = cx.toFixed(2) + ',' + cy.toFixed(2) + ',' + cz.toFixed(2);
            annoCell[1].innerHTML = diameter.toFixed(2);  
            annoCell[2].innerHTML = info;
        }
    }

    function annoListClean() {
        var annoTableRows = annotationTable.rows;
        while (annoTableRows.length > 1) {
            annotationTable.deleteRow(annoTableRows.length-1);
        }
    }

    function recvAnnotationList(tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader) {
        if (!socketClient.protocRoot) {
            console.log('null protocbuf.');
            return;
        }
        if (withHeader) {
            annotationListBuffer = new ArrayBuffer(dataLen + restDataLen);
        }

        var dstview = new Uint8Array(annotationListBuffer);
        var srcview = new Uint8Array(tcpBuffer, bufferOffset, dataLen);
        var cpSrc = annotationListBuffer.byteLength - (dataLen + restDataLen);
        for (var i = 0; i < dataLen; i++) {
            dstview[i+cpSrc] = srcview[i];
        }

        if (restDataLen <= 0) {
            var MsgAnnotationListType = socketClient.protocRoot.lookup('medical_imaging.MsgAnnotationList');
            if (!MsgAnnotationListType) {
                console.log('get annotation list message type failed.');
            }
            var annotationListView = new Uint8Array(annotationListBuffer);
            var annotationList = MsgAnnotationListType.decode(annotationListView);
            if (annotationList) {
                var listItems = annotationList.item;for (var i = 0; i < listItems.length; ++i) {
                    var id = listItems[i].id;
                    var info = listItems[i].info;
                    var row = listItems[i].row;
                    var status = listItems[i].status;
                    var cx = listItems[i].para0;
                    var cy = listItems[i].para1;
                    var cz = listItems[i].para2;
                    var diameter = listItems[i].para3;

                    if(status == 0) {//add
                        annoListAddRow(row, id, cx, cy, cz, diameter, info);
                    } else if (status == 1) {// delete
                        annoListDeleteRow(row);
                    } else if (status == 2) {// modifying
                        annoListModifyRow(row, cx, cy, cz, diameter, info);
                    }

                }
            }
        }
    }

    function cmdHandler(cmdID, cellID, opID, tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader) {
        switch (cmdID) {
            case COMMAND_ID_BE_SEND_IMAGE:
                cells[cellID].handleJpegBuffer(tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader);
                break;
            case COMMAND_ID_BE_SEND_NONE_IMAGE:
                cells[cellID].handleNongImgBuffer(tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader);
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
            case COMMAND_ID_BE_SEND_ANNOTATION_LIST:
                recvAnnotationList(tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader);
                break;
            default:
                break;
        }
    }

    function getProperCellSize() {
        var cellContainerW = document.getElementById('cell-container').offsetWidth;
        // var cellContainerH = document.getElementById('cell-container').offsetHeight;
        var navigatorHeight = document.getElementById('navigator-div').offsetHeight;
        var w = parseInt((cellContainerW - 40) / 2 + 0.5);
        var h = parseInt((window.innerHeight - navigatorHeight - 25) / 2 + 0.5);
        if (w%2 != 0) {
            w += 1;
        }
        if (h%2 != 0) {
            h += 1;
        }
        return {
            width: w,
            height: h
        };
    }

    function resize() {
        if (!revcBEReady) {
            return;
        }
        if (seriesUID === '') {
            return;
        }
        if (!socketClient.protocRoot) {
            console.log('null protocbuf.')
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

    function resizeCell(cellID, w, h) {
        if (!revcBEReady) {
            return;
        }
        if (seriesUID === '') {
            return;
        }
        if (!socketClient.protocRoot) {
            console.log('null protocbuf.')
            return;
        }
        cells[cellID].resize(w, h);

        var MsgResize = socketClient.protocRoot.lookup('medical_imaging.MsgResize');
        var msgResize = MsgResize.create();
        msgResize.cells.push({id: cellID, type: 1, direction: 0, width: w, height: h});
        var msgBuffer = MsgResize.encode(msgResize).finish();
        socketClient.sendData(COMMAND_ID_FE_OPERATION, OPERATION_ID_RESIZE, 0, msgBuffer.byteLength, msgBuffer);
    }

    function changeLayout(cellID) {
        if (layoutStatus == LAYOUT_1X1) {
            if (maxCellID == -1) {
                console.log('invalid max cell ID');
                reutrn;
            }
            document.getElementById('cell02-container').hidden = false;
            document.getElementById('cell13-container').hidden = false;
            document.getElementById('cell0-container').hidden = false;
            document.getElementById('cell1-container').hidden = false;
            document.getElementById('cell2-container').hidden = false;
            document.getElementById('cell3-container').hidden = false;
            
            var cellSize = getProperCellSize();
            resizeCell(maxCellID, cellSize.width, cellSize.height);
            maxCellID = -1;
            layoutStatus = LAYOUT_2X2;
        } else if (layoutStatus == LAYOUT_2X2) {
            switch (cellID) {
                case 0:
                    document.getElementById('cell13-container').hidden = true;
                    document.getElementById('cell2-container').hidden = true;
                    document.getElementById('cell0-container').hidden = false;
                    break
                case 1:
                    document.getElementById('cell02-container').hidden = true;
                    document.getElementById('cell3-container').hidden = true;
                    document.getElementById('cell1-container').hidden = false;
                    break;
                case 2:
                    document.getElementById('cell13-container').hidden = true;
                    document.getElementById('cell0-container').hidden = true;
                    document.getElementById('cell2-container').hidden = false;
                    break;
                case 3:
                    document.getElementById('cell02-container').hidden = true;
                    document.getElementById('cell1-container').hidden = true;
                    document.getElementById('cell3-container').hidden = false;
                    break;
            }
            var cellSize = getProperCellSize();
            resizeCell(cellID, cellSize.width*2, cellSize.height*2);
            maxCellID = cellID;
            layoutStatus = LAYOUT_1X1;
        }
    }

    function focusCell(cellID) {
        for (var i = 0; i < cells.length; ++i) {
            if (i == cellID) { 
                cells[i].mouseFocus = true;
                cellCanvases[i].style.border = '3px solid ' + '#6A5ACD';
            } else {
                cells[i].mouseFocus = false;
                cellCanvases[i].style.border = '3px solid ' + cells[i].borderColor;
            }
        }        
    }

    function resetPanel() {
        seriesUID = '';
    }

    function loadSeries(series) {
        seriesUID = series;
        if (!socketClient) {
            console.log('socket client is null.');
            return;
        }
        if (!socketClient.protocRoot) {
            console.log('null protocbuf.')
            return;
        }
        //recover cell layout
        document.getElementById('cell02-container').hidden = false;
        document.getElementById('cell13-container').hidden = false;
        document.getElementById('cell0-container').hidden = false;
        document.getElementById('cell1-container').hidden = false;
        document.getElementById('cell2-container').hidden = false;
        document.getElementById('cell3-container').hidden = false;

        //release previous cells
        if (cells && cells.length != 0) {
            for (var i = 0; i < cells.length; i++) {
                cells[i].release();
            }    
        }

        //create cells
        var cellSize = getProperCellSize();
        var w = cellSize.width;
        var h = cellSize.height;
        cells = [null, null, null, null];
        for (var i = 0; i < 4; i++) {
            var cellName = 'cell_' + i;
            var canvas = cellCanvases[i];
            var svg = cellSVGs[i];
            cells[i] = new Cell(cellName, i, canvas, svg, socketClient);
            cells[i].resize(cellSize.width, cellSize.height);
            cells[i].prepare();
            cells[i].mouseDoubleClickEvent = changeLayout;
            cells[i].mouseFocusEvent = focusCell;
        }

        //init default cell action
        for (var i = 0; i < 3; ++i) {
            cells[i].activeAction(ACTION_ID_MPR_PAGING, ACTION_ID_ZOOM, ACTION_ID_WINDOWING);
            //MPR add crosshair
            cells[i].crosshair = new Crosshair(cellSVGs[i], i, w/2, h/2,{a:2/w, b:0, c:1}, {a:0, b:2/h, c:1}, socketClient, 0);
        }
        cells[3].activeAction(ACTION_ID_ROTATE, ACTION_ID_ZOOM, ACTION_ID_WINDOWING);
        cells[3].crosshair = new Crosshair(cellSVGs[3], 3, 0, 0,{a:0, b:0, c:0}, {a:0, b:0, c:0}, socketClient, 1);



        //nofity BE
        var MsgInit = socketClient.protocRoot.lookup('medical_imaging.MsgInit');
        if (!MsgInit) {
            console.log('get init message type failed.');
            return;
        }
        var msgInit = MsgInit.create();
        if (!msgInit) {
            console.log('create init message failed.');
            return;
        }
        msgInit.seriesUid = seriesUID;
        msgInit.pid = 1000;
        msgInit.cells.push({id: 0, type: 1, direction: 0, width: w, height: h});
        msgInit.cells.push({id: 1, type: 1, direction: 1, width: w, height: h});
        msgInit.cells.push({id: 2, type: 1, direction: 2, width: w, height: h});
        msgInit.cells.push({id: 3, type: 2, direction: 0, width: w, height: h});
        var msgBuffer = MsgInit.encode(msgInit).finish();
        socketClient.sendData(COMMAND_ID_FE_OPERATION, OPERATION_ID_INIT, 0, msgBuffer.byteLength, msgBuffer);
    }

    function switchCommonTool(btnID) {
        document.getElementById('test-info').innerText = btnID;
        switch (btnID) {
            case 'common-tool-arrow':
                cells[0].activeAction(ACTION_ID_MPR_PAGING, ACTION_ID_ZOOM, ACTION_ID_WINDOWING);
                cells[1].activeAction(ACTION_ID_MPR_PAGING, ACTION_ID_ZOOM, ACTION_ID_WINDOWING);
                cells[2].activeAction(ACTION_ID_MPR_PAGING, ACTION_ID_ZOOM, ACTION_ID_WINDOWING);
                cells[3].activeAction(ACTION_ID_ROTATE, ACTION_ID_ZOOM, ACTION_ID_WINDOWING);
                break;
            case 'common-tool-zoom':
                for (var i = 0; i < cells.length; ++i) {
                    cells[i].activeAction(ACTION_ID_ZOOM, ACTION_ID_ZOOM, ACTION_ID_WINDOWING);
                }
                break;
            case 'common-tool-pan':
                for (var i = 0; i < cells.length; ++i) {
                    cells[i].activeAction(ACTION_ID_PAN, ACTION_ID_ZOOM, ACTION_ID_WINDOWING);
                }
                break;
            case 'common-tool-rotate':
                cells[0].activeAction(ACTION_ID_MPR_PAGING, ACTION_ID_ZOOM, ACTION_ID_WINDOWING);
                cells[1].activeAction(ACTION_ID_MPR_PAGING, ACTION_ID_ZOOM, ACTION_ID_WINDOWING);
                cells[2].activeAction(ACTION_ID_MPR_PAGING, ACTION_ID_ZOOM, ACTION_ID_WINDOWING);
                cells[3].activeAction(ACTION_ID_ROTATE, ACTION_ID_ZOOM, ACTION_ID_WINDOWING);
                break;
            case 'common-tool-windowing':
                for (var i = 0; i < cells.length; ++i) {
                    cells[i].activeAction(ACTION_ID_WINDOWING, ACTION_ID_ZOOM, ACTION_ID_WINDOWING);
                }
                break;
            case 'common-tool-annotation':
                cells[0].activeAction(ACTION_ID_MRP_ANNOTATION, ACTION_ID_ZOOM, ACTION_ID_WINDOWING);
                cells[1].activeAction(ACTION_ID_MRP_ANNOTATION, ACTION_ID_ZOOM, ACTION_ID_WINDOWING);
                cells[2].activeAction(ACTION_ID_MRP_ANNOTATION, ACTION_ID_ZOOM, ACTION_ID_WINDOWING);
                break;
            default:
                // TODO ERR
                break;
        }
    }

    function searchWorklist() {
        if (!revcBEReady) {
            console.log('BE not ready!');
            alert('BE not ready!');
            return;
        }
        socketClient.sendData(COMMAND_ID_FE_SEARCH_WORKLIST, 0, 0, null);
    }

    function playVR() {
        socketClient.sendData(COMMAND_ID_FE_VR_PLAY, 0, 3, null);
    }

    function prepare() {
        // disable the annoying context menu triggered by the right button
        document.addEventListener('contextmenu', event => event.preventDefault());
        
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
        var searchWorklistBtn = document.getElementById('btn-search-worklist');
        if (searchWorklist) {
            searchWorklistBtn.onclick = function(event) {
                searchWorklist();
            };
        } else {
            console.log('get searchBtn node failed.');
        }
        
        var loadSeriesBtn = document.getElementById('btn-load-series');
        if (loadSeriesBtn) {
            loadSeriesBtn.onclick = function(event) {
                var series = $('#table tbody tr.success td:nth-child(3)').html();
                if (!series) {
                    alert('please choose one series.');
                    reutrn;
                }
                document.getElementById('worklist-div').hidden = true;
                document.getElementById('review-div').hidden = false;
                annoListClean();
                loadSeries(series);
            };
        } else {
            console.log('get loadBtn node failed.');
        }

        var comToolsDiv = document.getElementById('common-tools');
        if (comToolsDiv) {
            var comToolsBtns = comToolsDiv.getElementsByTagName('button');
            for (var i = 0; i < comToolsBtns.length; ++i) {
                comToolsBtns[i].onclick = function(event) {
                    $(this).addClass('btn-primary').siblings().removeClass('btn-primary');
                    switchCommonTool(this.id);
                };
            }
        } else {
            console.log('get common-tools failed.');
        }
        $('#common-tool-arrow').addClass('btn-primary').siblings().removeClass('btn-primary');

        var playVRBtn = document.getElementById('btn-play-vr');
        if (playVRBtn) {
            playVRBtn.onclick = function() {
                playVR();
            };
        } else {
            console.log('get btn-play-vr node failed.');
        }

        annotationTable = document.getElementById("annotation-table");

        var deleteAnnotationBtn = document.getElementById('btn-delete-annotation');
        if (deleteAnnotationBtn) {
            deleteAnnotationBtn.onclick = function(event) {
                var choosedItem = $('#annotation-table tr.success');
                if (choosedItem.length > 0) {
                    var id = choosedItem.attr('id')
                    if (id) {
                        sendAnnotationMSG(0, 0, id, 1, false, 0, 0, 0, socketClient);//Delete msg
                    }
                }

            };
        }

        var mprMaskOverlayFunc = function(event) {
            var flag = 1;
            if( document.getElementById('cbox-overlay-annotation') ) {
                flag = document.getElementById('cbox-overlay-annotation').checked ? 1 : 0;
            }
            var opacity = 0.5;
            if (document.getElementById('range-mpr-overlay-opacity')) {
                opacity = document.getElementById('range-mpr-overlay-opacity').value;
            } 
            
            if (!socketClient.protocRoot) {
                console.log('null protobuf.');
                return;
            }
            var MsgMPRMaskOverlayType = socketClient.protocRoot.lookup('medical_imaging.MsgMPRMaskOverlay');
            if (!MsgMPRMaskOverlayType) {
                console.log('get MsgMPRMaskOverlay type failed.');
                return;
            }
            var msg = MsgMPRMaskOverlayType.create({flag:flag, opacity:opacity});
            if (!msg) {
                console.log('create mpr mask overlay message failed.');
                return;
            }
            var msgBuffer = MsgMPRMaskOverlayType.encode(msg).finish();
            if (!msgBuffer) {
                console.log('encode mpr mask overlay message failed.');
            }
            socketClient.sendData(COMMAND_ID_FE_OPERATION, OPERATION_ID_MPR_MASK_OVERLAY, 0, msgBuffer.byteLength, msgBuffer);
        }

        var annotationOverlayCBox = document.getElementById('cbox-overlay-annotation');
        if (annotationOverlayCBox) {
            annotationOverlayCBox.defaultChecked = true;
            annotationOverlayCBox.onclick = mprMaskOverlayFunc;
        }

        var mprMaskOverlayOpacityRange = document.getElementById('range-mpr-overlay-opacity');
        if (mprMaskOverlayOpacityRange) {
            mprMaskOverlayOpacityRange.max = 1;
            mprMaskOverlayOpacityRange.min = 0;
            mprMaskOverlayOpacityRange.step = 0.025;
            mprMaskOverlayOpacityRange.defaultValue = 0.5;
            mprMaskOverlayOpacityRange.onchange = mprMaskOverlayFunc;
            mprMaskOverlayOpacityRange.oninput = mprMaskOverlayFunc;
        }

        var goBackImg = document.getElementById('btn-back-worklist');
        if(goBackImg)
        {
            goBackImg.onclick = function(event) {
                document.getElementById('worklist-div').hidden = false;
                document.getElementById('review-div').hidden = true;
            }
        }

        var layout1x1Btn = document.getElementById('btn-layout1x1');
        if (layout1x1Btn) {
            layout1x1Btn.onclick = function(event) {
                if (layoutStatus == LAYOUT_2X2) {
                    var focusCellID = 0;
                    for (var i = 0; i< cells.length; ++i) {
                        if (cells[i].mouseFocus) {
                            focusCellID = i;
                            break;
                        }
                    }
                    changeLayout(focusCellID);
                }
            }
        }

        var layout2x2Btn = document.getElementById('btn-layout2x2');
        if (layout2x2Btn) {
            layout2x2Btn.onclick = function(event) {
                if (layoutStatus == LAYOUT_1X1 && maxCellID != -1) {
                    changeLayout(maxCellID);
                }
            }
        }

        var switchPresetWLFunc = function(obj) {
            document.getElementById('btn-preset-wl').innerHTML = obj.innerHTML + '<span class="caret"></span>';
            if (!socketClient.protocRoot) {
                console.log('null protobuf.');
                return;
            }
            var MsgStringType = socketClient.protocRoot.lookup('medical_imaging.MsgString');
            if (!MsgStringType) {
                console.log('get MsgMsgStringType type failed.');
                return;
            }
            var msg = MsgStringType.create({context:obj.innerHTML});
            if (!msg) {
                console.log('create switch preset WL message failed.');
                return;
            }
            var msgBuffer = MsgStringType.encode(msg).finish();
            if (!msgBuffer) {
                console.log('encode switch preset WL message failed.');
            }
            socketClient.sendData(COMMAND_ID_FE_OPERATION, OPERATION_ID_SWITCH_PRESET_WINDOWING, 0, msgBuffer.byteLength, msgBuffer);
        }
        document.getElementById('a-preset-wl-abdomen').onclick = function(event) {switchPresetWLFunc(this);return false;}
        document.getElementById('a-preset-wl-lung').onclick = function(event) {switchPresetWLFunc(this);return false;}
        document.getElementById('a-preset-wl-brain').onclick = function(event) {switchPresetWLFunc(this);return false;}
        document.getElementById('a-preset-wl-angio').onclick = function(event) {switchPresetWLFunc(this);return false;}
        document.getElementById('a-preset-wl-bone').onclick = function(event) {switchPresetWLFunc(this);return false;}
        document.getElementById('a-preset-wl-chest').onclick = function(event) {switchPresetWLFunc(this);return false;}

        var crosshairContinuousCBox = document.getElementById('cbox-crosshair-continuous');
        if (crosshairContinuousCBox) {
            crosshairContinuousCBox.defaultChecked = false;
            crosshairContinuousCBox.onclick = function(event) {
                for (var i = 0; i< 4; ++i) {
                    if (cells[i].crosshair){
                        cells[i].crosshair.crossContinuous = crosshairContinuousCBox.checked;
                    }
                }
            }
        }

        var crosshairVisibleCBox = document.getElementById('cbox-crosshair-visible');
        if (crosshairVisibleCBox) {
            crosshairVisibleCBox.defaultChecked = true;
            crosshairVisibleCBox.onclick = function(event) {
                for (var i = 0; i< 4; ++i) {
                    if (cells[i].crosshair){
                        cells[i].crosshair.visible(crosshairVisibleCBox.checked);
                    }
                }
            }
        }
        // disable back button
        history.pushState(null, null, document.URL);
        window.addEventListener('popstate', function() {
          history.pushState(null, null, document.URL);
        });


        $('#modal-preset-vrt-browser').draggable({
            handle: '.modal-header'
        });
        
        var switchVRTFunc = function(context) {
            if (!socketClient.protocRoot) {
                console.log('null protobuf.');
                return;
            }
            var MsgStringType = socketClient.protocRoot.lookup('medical_imaging.MsgString');
            if (!MsgStringType) {
                console.log('get MsgMsgStringType type failed.');
                return;
            }
            var msg = MsgStringType.create({context:context});
            if (!msg) {
                console.log('create switch vrt message failed.');
                return;
            }
            var msgBuffer = MsgStringType.encode(msg).finish();
            if (!msgBuffer) {
                console.log('encode switch vrt message failed.');
            }
            socketClient.sendData(COMMAND_ID_FE_OPERATION, OPERATION_ID_SWITCH_PRESET_VRT, 0, msgBuffer.byteLength, msgBuffer);
        }
        document.getElementById('img-preset-vrt-cta').onclick = function(event) {switchVRTFunc('cta');}
        document.getElementById('img-preset-vrt-lung-glass').onclick = function(event) {switchVRTFunc('lung-glass');}

        var loginBtn = document.getElementById('btn-login');
        if (loginBtn) {
            loginBtn.onclick = function(event) {
                window.location.href = '/login';
            }
        }

        // register window quit linsener
        window.onbeforeunload = function(event) {
            logout();
        }
        window.onresize = function() {
            resize()
        };
        var username = document.getElementById('username').innerHTML;
        login();
    }

    prepare();
})();