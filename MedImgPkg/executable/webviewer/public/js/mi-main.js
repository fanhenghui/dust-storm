
(function() {
    let socket = null;
    let serverIP = '';
    let seriesUID = '';
    let cellCanvases = null;
    let cellSVGs = null;
    let cells = null;
    let protocRoot = null;
    let worklistBuffer = null;
    let socketClient = null;
    let revcBEReady = false;
    
    let annotationListBuffer = null;
    let annotationTable = null;
    
    //layout parameter
    const LAYOUT_1X1 = '1x1';
    const LAYOUT_2X2 = '2x2';
    let layoutStatus = LAYOUT_2X2;
    let maxCellID = -1;

    function getUserID(userName) {
        return userName + '|' + new Date().getTime() + Math.floor(Math.random() * 173 + 511);
    }

    function login() {
        socket = io.connect(serverIP);
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

    let heartbeatCount = 1;
    let lastheartbearCount = 0;
    const HEARTBEAT_INTERVAL = 5 * 1000 + 1000;
    function keepHeartbeat() {
        socketClient.heartbeat();
        heartbeatCount += 1;
    }
    function checkHeartbeat() {
        if (!revcBEReady) {
            return;
        }
        if (heartbeatCount - lastheartbearCount > 0) {
            lastheartbearCount = heartbeatCount;
        } else {
            //return to login
            console.log('heart dead. return to login.');
            window.location.href = '/login';
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

        let dstview = new Uint8Array(worklistBuffer);
        let srcview = new Uint8Array(tcpBuffer, bufferOffset, dataLen);
        let cpSrc = worklistBuffer.byteLength - (dataLen + restDataLen);
        for (let i = 0; i < dataLen; i++) {
            dstview[i+cpSrc] = srcview[i];
        }

        if (restDataLen <= 0) {
            console.log('recv worklist.');
            let MsgWorklistType = socketClient.protocRoot.lookup('medical_imaging.MsgWorklist');
            if (!MsgWorklistType) {
                console.log('get worklist message type failed.');
            }
            let worklistView = new Uint8Array(worklistBuffer);
            let message = MsgWorklistType.decode(worklistView);
            let obj = MsgWorklistType.toObject(message, {
                patient_name: String,
                patient_id: String,
                series_uid: String,
                imaging_modality: String
            }).items;
            let tbody = document.getElementById('worklist');
            tbody.innerHTML = '';
            for (let i = 0; i < obj.length; i++) {
                let tr = '<tr>';
                for (let propt in obj[i]) {
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
        let annoTableRows = annotationTable.rows;
        if (annoTableRows.length > row + 1) {
            annotationTable.deleteRow(row + 1);
        }
    }

    function annoListAddRow(row,id,cx,cy,cz,diameter,info) {
        if (document.getElementById(id)) {
            console.log('add repeated row');
            return;
        }
        let rowItem = annotationTable.insertRow(row + 1);
        $(rowItem).click(function(event) {
            $(this).addClass('success').siblings().removeClass('success');
            //send focus annotation message
            let anno_id = $(this).attr('id');
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
        let annoTableRows = annotationTable.rows;
        if (annoTableRows.length > row + 1) {
            let annoCell = annoTableRows[row + 1].cells;
            annoCell[0].innerHTML = cx.toFixed(2) + ',' + cy.toFixed(2) + ',' + cz.toFixed(2);
            annoCell[1].innerHTML = diameter.toFixed(2);  
            annoCell[2].innerHTML = info;
        }
    }

    function annoListClean() {
        let annoTableRows = annotationTable.rows;
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

        let dstview = new Uint8Array(annotationListBuffer);
        let srcview = new Uint8Array(tcpBuffer, bufferOffset, dataLen);
        let cpSrc = annotationListBuffer.byteLength - (dataLen + restDataLen);
        for (let i = 0; i < dataLen; i++) {
            dstview[i+cpSrc] = srcview[i];
        }

        if (restDataLen <= 0) {
            let MsgAnnotationListType = socketClient.protocRoot.lookup('medical_imaging.MsgAnnotationList');
            if (!MsgAnnotationListType) {
                console.log('get annotation list message type failed.');
            }
            let annotationListView = new Uint8Array(annotationListBuffer);
            let annotationList = MsgAnnotationListType.decode(annotationListView);
            if (annotationList) {
                let listItems = annotationList.item;for (let i = 0; i < listItems.length; ++i) {
                    let id = listItems[i].id;
                    let info = listItems[i].info;
                    let row = listItems[i].row;
                    let status = listItems[i].status;
                    let cx = listItems[i].para0;
                    let cy = listItems[i].para1;
                    let cz = listItems[i].para2;
                    let diameter = listItems[i].para3;

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
                keepHeartbeat();
                break;
            case COMMAND_ID_BE_SEND_ANNOTATION_LIST:
                recvAnnotationList(tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader);
                break;
            default:
                break;
        }
    }

    function getProperCellSize() {
        let cellContainerW = document.getElementById('cell-container').offsetWidth;
        // let cellContainerH = document.getElementById('cell-container').offsetHeight;
        let navigatorHeight = document.getElementById('navigator-div').offsetHeight;
        let w = parseInt((cellContainerW - 40) / 2 + 0.5);
        let h = parseInt((window.innerHeight - navigatorHeight - 25) / 2 + 0.5);
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
        let cellSize = getProperCellSize();
        let w = cellSize.width;
        let h = cellSize.height;
        for (let i = 0; i < cells.length; i++) {
            cells[i].resize(w, h);
        }

        let MsgResize = socketClient.protocRoot.lookup('medical_imaging.MsgResize');
        let msgResize = MsgResize.create();
        msgResize.cells.push({id: 0, type: 1, direction: 0, width: w, height: h});
        msgResize.cells.push({id: 1, type: 1, direction: 1, width: w, height: h});
        msgResize.cells.push({id: 2, type: 1, direction: 2, width: w, height: h});
        msgResize.cells.push({id: 3, type: 2, direction: 0, width: w, height: h});
        let msgBuffer = MsgResize.encode(msgResize).finish();
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

        let MsgResize = socketClient.protocRoot.lookup('medical_imaging.MsgResize');
        let msgResize = MsgResize.create();
        msgResize.cells.push({id: cellID, type: 1, direction: 0, width: w, height: h});
        let msgBuffer = MsgResize.encode(msgResize).finish();
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
            
            let cellSize = getProperCellSize();
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
            let cellSize = getProperCellSize();
            resizeCell(cellID, cellSize.width*2, cellSize.height*2);
            maxCellID = cellID;
            layoutStatus = LAYOUT_1X1;
        }
    }

    function focusCell(cellID) {
        for (let i = 0; i < cells.length; ++i) {
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
            for (let i = 0; i < cells.length; i++) {
                cells[i].release();
            }    
        }

        //create cells
        let cellSize = getProperCellSize();
        let w = cellSize.width;
        let h = cellSize.height;
        cells = [null, null, null, null];
        for (let i = 0; i < 4; i++) {
            let cellName = 'cell_' + i;
            let canvas = cellCanvases[i];
            let svg = cellSVGs[i];
            cells[i] = new Cell(cellName, i, canvas, svg, socketClient);
            cells[i].resize(cellSize.width, cellSize.height);
            cells[i].prepare();
            cells[i].mouseDoubleClickEvent = changeLayout;
            cells[i].mouseFocusEvent = focusCell;
            cells[i].mouseActionAnnotation.upCallback = function() {
                let btnArrow = document.getElementById('common-tool-arrow');
                $(btnArrow).addClass('btn-primary').siblings().removeClass('btn-primary');
                switchCommonTool('common-tool-arrow');
            };
        }

        //init default cell action
        for (let i = 0; i < 3; ++i) {
            cells[i].activeAction(ACTION_ID_MPR_PAGING, ACTION_ID_ZOOM, ACTION_ID_WINDOWING, ACTION_ID_PAN);
            //MPR add crosshair
            cells[i].crosshair = new Crosshair(cellSVGs[i], i, w/2, h/2,{a:2/w, b:0, c:1}, {a:0, b:2/h, c:1}, socketClient, 0);
            //MPR add ruler
            cells[i].ruler = new VerticalRuler(cellSVGs[i], i);
        }
        cells[3].activeAction(ACTION_ID_ROTATE, ACTION_ID_ZOOM, ACTION_ID_WINDOWING, ACTION_ID_PAN);
        cells[3].crosshair = new Crosshair(cellSVGs[3], 3, 0, 0,{a:0, b:0, c:0}, {a:0, b:0, c:0}, socketClient, 1);



        //nofity BE
        let MsgInit = socketClient.protocRoot.lookup('medical_imaging.MsgInit');
        if (!MsgInit) {
            console.log('get init message type failed.');
            return;
        }
        let msgInit = MsgInit.create();
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
        let msgBuffer = MsgInit.encode(msgInit).finish();
        socketClient.sendData(COMMAND_ID_FE_OPERATION, OPERATION_ID_INIT, 0, msgBuffer.byteLength, msgBuffer);
    }

    function switchCommonTool(btnID) {
        document.getElementById('test-info').innerText = btnID;
        switch (btnID) {
            case 'common-tool-arrow':
                cells[0].activeAction(ACTION_ID_MPR_PAGING, ACTION_ID_ZOOM, ACTION_ID_WINDOWING, ACTION_ID_PAN);
                cells[1].activeAction(ACTION_ID_MPR_PAGING, ACTION_ID_ZOOM, ACTION_ID_WINDOWING, ACTION_ID_PAN);
                cells[2].activeAction(ACTION_ID_MPR_PAGING, ACTION_ID_ZOOM, ACTION_ID_WINDOWING, ACTION_ID_PAN);
                cells[3].activeAction(ACTION_ID_ROTATE, ACTION_ID_ZOOM, ACTION_ID_WINDOWING, ACTION_ID_PAN);
                break;
            case 'common-tool-zoom':
                for (let i = 0; i < cells.length; ++i) {
                    cells[i].activeAction(ACTION_ID_ZOOM, ACTION_ID_ZOOM, ACTION_ID_WINDOWING, ACTION_ID_PAN);
                }
                break;
            case 'common-tool-pan':
                for (let i = 0; i < cells.length; ++i) {
                    cells[i].activeAction(ACTION_ID_PAN, ACTION_ID_ZOOM, ACTION_ID_WINDOWING, ACTION_ID_PAN);
                }
                break;
            case 'common-tool-rotate':
                cells[0].activeAction(ACTION_ID_MPR_PAGING, ACTION_ID_ZOOM, ACTION_ID_WINDOWING, ACTION_ID_PAN);
                cells[1].activeAction(ACTION_ID_MPR_PAGING, ACTION_ID_ZOOM, ACTION_ID_WINDOWING, ACTION_ID_PAN);
                cells[2].activeAction(ACTION_ID_MPR_PAGING, ACTION_ID_ZOOM, ACTION_ID_WINDOWING, ACTION_ID_PAN);
                cells[3].activeAction(ACTION_ID_ROTATE, ACTION_ID_ZOOM, ACTION_ID_WINDOWING, ACTION_ID_PAN);
                break;
            case 'common-tool-windowing':
                for (let i = 0; i < cells.length; ++i) {
                    cells[i].activeAction(ACTION_ID_WINDOWING, ACTION_ID_ZOOM, ACTION_ID_WINDOWING, ACTION_ID_PAN);
                }
                break;
            case 'common-tool-annotation':
                cells[0].activeAction(ACTION_ID_MRP_ANNOTATION, ACTION_ID_ZOOM, ACTION_ID_WINDOWING, ACTION_ID_PAN);
                cells[1].activeAction(ACTION_ID_MRP_ANNOTATION, ACTION_ID_ZOOM, ACTION_ID_WINDOWING, ACTION_ID_PAN);
                cells[2].activeAction(ACTION_ID_MRP_ANNOTATION, ACTION_ID_ZOOM, ACTION_ID_WINDOWING, ACTION_ID_PAN);
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
        document.oncontextmenu = function(event) {
            event.preventDefault();
        }
        
        // Create cell object
        let cellContainer = document.getElementById('cell-container');
        cellCanvases = [
            document.getElementById('canvas0'), document.getElementById('canvas1'),
            document.getElementById('canvas2'), document.getElementById('canvas3')
        ];
        cellSVGs = [
            document.getElementById('svg0'), document.getElementById('svg1'),
            document.getElementById('svg2'), document.getElementById('svg3')
        ];

        // register button event
        let searchWorklistBtn = document.getElementById('btn-search-worklist');
        if (searchWorklist) {
            searchWorklistBtn.onclick = function(event) {
                searchWorklist();
            };
        } else {
            console.log('get searchBtn node failed.');
        }
        
        let loadSeriesBtn = document.getElementById('btn-load-series');
        if (loadSeriesBtn) {
            loadSeriesBtn.onclick = function(event) {
                let series = $('#table tbody tr.success td:nth-child(3)').html();
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

        let comToolsDiv = document.getElementById('common-tools');
        if (comToolsDiv) {
            let comToolsBtns = comToolsDiv.getElementsByTagName('button');
            for (let i = 0; i < comToolsBtns.length; ++i) {
                comToolsBtns[i].onclick = function(event) {
                    $(this).addClass('btn-primary').siblings().removeClass('btn-primary');
                    switchCommonTool(this.id);
                };
            }
        } else {
            console.log('get common-tools failed.');
        }
        $('#common-tool-arrow').addClass('btn-primary').siblings().removeClass('btn-primary');

        let playVRBtn = document.getElementById('btn-play-vr');
        if (playVRBtn) {
            playVRBtn.onclick = function() {
                playVR();
            };
        } else {
            console.log('get btn-play-vr node failed.');
        }

        annotationTable = document.getElementById("annotation-table");

        let deleteAnnotationBtn = document.getElementById('btn-delete-annotation');
        if (deleteAnnotationBtn) {
            deleteAnnotationBtn.onclick = function(event) {
                let choosedItem = $('#annotation-table tr.success');
                if (choosedItem.length > 0) {
                    let id = choosedItem.attr('id')
                    if (id) {
                        sendAnnotationMSG(0, 0, id, 1, false, 0, 0, 0, socketClient);//Delete msg
                    }
                }

            };
        }

        let mprMaskOverlayFunc = function(event) {
            let flag = 1;
            if( document.getElementById('cbox-overlay-annotation') ) {
                flag = document.getElementById('cbox-overlay-annotation').checked ? 1 : 0;
            }
            let opacity = 0.5;
            if (document.getElementById('range-mpr-overlay-opacity')) {
                opacity = document.getElementById('range-mpr-overlay-opacity').value;
            } 
            
            if (!socketClient.protocRoot) {
                console.log('null protobuf.');
                return;
            }
            let MsgMPRMaskOverlayType = socketClient.protocRoot.lookup('medical_imaging.MsgMPRMaskOverlay');
            if (!MsgMPRMaskOverlayType) {
                console.log('get MsgMPRMaskOverlay type failed.');
                return;
            }
            let msg = MsgMPRMaskOverlayType.create({flag:flag, opacity:opacity});
            if (!msg) {
                console.log('create mpr mask overlay message failed.');
                return;
            }
            let msgBuffer = MsgMPRMaskOverlayType.encode(msg).finish();
            if (!msgBuffer) {
                console.log('encode mpr mask overlay message failed.');
            }
            socketClient.sendData(COMMAND_ID_FE_OPERATION, OPERATION_ID_MPR_MASK_OVERLAY, 0, msgBuffer.byteLength, msgBuffer);
        }

        let annotationOverlayCBox = document.getElementById('cbox-overlay-annotation');
        if (annotationOverlayCBox) {
            annotationOverlayCBox.defaultChecked = true;
            annotationOverlayCBox.onclick = mprMaskOverlayFunc;
        }

        let mprMaskOverlayOpacityRange = document.getElementById('range-mpr-overlay-opacity');
        if (mprMaskOverlayOpacityRange) {
            mprMaskOverlayOpacityRange.max = 1;
            mprMaskOverlayOpacityRange.min = 0;
            mprMaskOverlayOpacityRange.step = 0.025;
            mprMaskOverlayOpacityRange.defaultValue = 0.5;
            mprMaskOverlayOpacityRange.onchange = mprMaskOverlayFunc;
            mprMaskOverlayOpacityRange.oninput = mprMaskOverlayFunc;
        }

        let goBackImg = document.getElementById('btn-back-worklist');
        if(goBackImg)
        {
            goBackImg.onclick = function(event) {
                document.getElementById('worklist-div').hidden = false;
                document.getElementById('review-div').hidden = true;
            }
        }

        let layout1x1Btn = document.getElementById('btn-layout1x1');
        if (layout1x1Btn) {
            layout1x1Btn.onclick = function(event) {
                if (layoutStatus == LAYOUT_2X2) {
                    let focusCellID = 0;
                    for (let i = 0; i< cells.length; ++i) {
                        if (cells[i].mouseFocus) {
                            focusCellID = i;
                            break;
                        }
                    }
                    changeLayout(focusCellID);
                }
            }
        }

        let layout2x2Btn = document.getElementById('btn-layout2x2');
        if (layout2x2Btn) {
            layout2x2Btn.onclick = function(event) {
                if (layoutStatus == LAYOUT_1X1 && maxCellID != -1) {
                    changeLayout(maxCellID);
                }
            }
        }

        let switchPresetWLFunc = function(obj) {
            document.getElementById('btn-preset-wl').innerHTML = obj.innerHTML + '<span class="caret"></span>';
            if (!socketClient.protocRoot) {
                console.log('null protobuf.');
                return;
            }
            let MsgStringType = socketClient.protocRoot.lookup('medical_imaging.MsgString');
            if (!MsgStringType) {
                console.log('get MsgMsgStringType type failed.');
                return;
            }
            let msg = MsgStringType.create({context:obj.innerHTML});
            if (!msg) {
                console.log('create switch preset WL message failed.');
                return;
            }
            let msgBuffer = MsgStringType.encode(msg).finish();
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

        let crosshairContinuousCBox = document.getElementById('cbox-crosshair-continuous');
        if (crosshairContinuousCBox) {
            crosshairContinuousCBox.defaultChecked = false;
            crosshairContinuousCBox.onclick = function(event) {
                for (let i = 0; i< 4; ++i) {
                    if (cells[i].crosshair){
                        cells[i].crosshair.crossContinuous = crosshairContinuousCBox.checked;
                    }
                }
            }
        }

        let crosshairVisibleCBox = document.getElementById('cbox-crosshair-visible');
        if (crosshairVisibleCBox) {
            crosshairVisibleCBox.defaultChecked = true;
            crosshairVisibleCBox.onclick = function(event) {
                for (let i = 0; i< 4; ++i) {
                    if (cells[i].crosshair){
                        cells[i].crosshair.visible(crosshairVisibleCBox.checked);
                    }
                }
            }
        }
        // disable back button
        history.pushState(null, null, document.URL);
        window.onpopstate =  function(event) {
          history.pushState(null, null, document.URL);
        };


        $('#modal-preset-vrt-browser').draggable({
            handle: '.modal-header'
        });
        
        let switchVRTFunc = function(context) {
            if (!socketClient.protocRoot) {
                console.log('null protobuf.');
                return;
            }
            let MsgStringType = socketClient.protocRoot.lookup('medical_imaging.MsgString');
            if (!MsgStringType) {
                console.log('get MsgMsgStringType type failed.');
                return;
            }
            let msg = MsgStringType.create({context:context});
            if (!msg) {
                console.log('create switch vrt message failed.');
                return;
            }
            let msgBuffer = MsgStringType.encode(msg).finish();
            if (!msgBuffer) {
                console.log('encode switch vrt message failed.');
            }
            socketClient.sendData(COMMAND_ID_FE_OPERATION, OPERATION_ID_SWITCH_PRESET_VRT, 0, msgBuffer.byteLength, msgBuffer);
        }

        let presetVRTTable = document.getElementById('table-preset-vrt');
        if (presetVRTTable) {
            let vrtRows = presetVRTTable.rows;
            for (let i = 0; i < vrtRows.length; ++i) {
                vrtCells = vrtRows[i].cells;
                for (let j = 0; j < vrtCells.length; ++j) {
                    vrtCells[j].onclick = (function() {
                        switchVRTFunc($(this).attr('id'));
                    }).bind(vrtCells[j]);    
                }
            }
        }
        // document.getElementById('img-preset-vrt-cta').onclick = function(event) {switchVRTFunc('ct_cta');}
        // document.getElementById('img-preset-vrt-lung-glass').onclick = function(event) {switchVRTFunc('ct_lung_glass');}

        ///For testing
        const TEST_INTERVAL = MOUSE_MSG_INTERVAL;
        function LeftMove(cellID, moveBack, moveX, step) {
            let x = moveBack ? 200 - step : 200 + step;
            if (moveX) {
                cells[cellID].mouseBtn = BTN_LEFT;
                cells[cellID].mouseStatus = BTN_DOWN;
                cells[cellID].mouseActionCommon.mouseDown(cells[cellID].mouseBtn, cells[cellID].mouseStatus, 200, 200, 200, 200, cells[cellID]);
                cells[cellID].mouseActionCommon.mouseMove(cells[cellID].mouseBtn, cells[cellID].mouseStatus, 200, 200, x, 200, cells[cellID]);
                cells[cellID].mouseActionCommon.mouseUp(cells[cellID].mouseBtn, cells[cellID].mouseStatus, x, 200, 200, 200, cells[cellID]);
                cells[cellID].mouseBtn = BTN_NONE;
                cells[cellID].mouseStatus = BTN_UP;
            } else {
                cells[cellID].mouseBtn = BTN_LEFT;
                cells[cellID].mouseStatus = BTN_DOWN;
                cells[cellID].mouseActionCommon.mouseDown(cells[cellID].mouseBtn, cells[cellID].mouseStatus, 200, 200, 200, 200, cells[cellID]);
                cells[cellID].mouseActionCommon.mouseMove(cells[cellID].mouseBtn, cells[cellID].mouseStatus, 200, 200, 200, x, cells[cellID]);
                cells[cellID].mouseActionCommon.mouseUp(cells[cellID].mouseBtn, cells[cellID].mouseStatus, 200, x, 200, 200, cells[cellID]);
                cells[cellID].mouseBtn = BTN_NONE;
                cells[cellID].mouseStatus = BTN_UP;
            }
        };

        const ROLL_INTERVAL = 200;
        tik = 0;
        roolStatus = false;
        function LeftMoveRoll(cellID, moveX, step) {
            tik += 1;
            if (tik > ROLL_INTERVAL) {
                tik = 0;
                roolStatus = !roolStatus;
            }

            if (roolStatus) {
                LeftMove(cellID, false, moveX, step);
            } else {
                LeftMove(cellID, true, moveX, step);
            }
        }

        testBtnStatus0 = false;
        testBtnFunc0 = null;
        let testBtn = document.getElementById('btn-test-0');
        if (testBtn) {
            testBtn.onclick = function(event) {
                if (!testBtnStatus0) {
                    testBtnStatus0 = true;
                    $('#btn-test-0').addClass('btn-primary');
                    if (testBtnFunc0 == null) {
                        testBtnFunc0 = setInterval(function() {
                            LeftMoveRoll(0, false, 2);
                        }, TEST_INTERVAL);
                    }
                } else {
                    $('#btn-test-0').removeClass('btn-primary');
                    testBtnStatus0 = false;
                    if (testBtnFunc0) {
                        clearInterval(testBtnFunc0);
                        testBtnFunc0 = null;
                    }
                }
            };
        }

        testBtnStatus1 = false;
        testBtnFunc1 = null;
        let testBtn1 = document.getElementById('btn-test-1');
        if (testBtn1) {
            testBtn1.onclick = function(event) {
                if (!testBtnStatus1) {
                    testBtnStatus1 = true;
                    $('#btn-test-1').addClass('btn-primary');
                    if (testBtnFunc1 == null) {
                        testBtnFunc1 = setInterval(function() {
                            LeftMove(3, false, true, 8);
                        }, TEST_INTERVAL);
                    }
                } else {
                    $('#btn-test-1').removeClass('btn-primary');
                    testBtnStatus1 = false;
                    if (testBtnFunc1) {
                        clearInterval(testBtnFunc1);
                        testBtnFunc1 = null;
                    }
                }
            };
        }

        let loginBtn = document.getElementById('btn-login-0');
        if (loginBtn) {
            loginBtn.onclick = function(event) {
                window.location.href = '/login';
            }
        }
        loginBtn = document.getElementById('btn-login-1');
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
        let username = document.getElementById('username').innerHTML;
        serverIP = document.getElementById('serverip').innerHTML;
        console.log('server ip: ' + serverIP);
        login();

        //trigger on heartbeat
        let checkHeartbearFunc = setInterval(function() {
            checkHeartbeat();
        }, HEARTBEAT_INTERVAL);
    }

    prepare();
})();