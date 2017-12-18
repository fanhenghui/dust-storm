
(function() {
    let socket = null;
    let seriesUID = '';
    let revcBEReady = false;

    let cellCanvases = null;
    let cellSVGs = null;
    let cells = null;
    let socketClient = null;
    let annotationList = null;
    
    //layout parameter
    const LAYOUT_1X1 = '1x1';
    const LAYOUT_2X2 = '2x2';
    let layoutStatus = LAYOUT_2X2;
    let maxCellID = -1;

    function getUserID(userName) {
        return userName + '|' + new Date().getTime() + Math.floor(Math.random() * 173 + 511);
    }

    function login() {
        //socket = io.connect("ws://172.23.237.144:8081");//标准写法，nginx代理的时候填写nginx代理服务器的地址
        socket = io(); //这样写貌似也可以
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
    const HEARTBEAT_INTERVAL = 12 * 1000; // 11s
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
            let subpage = document.getElementById('subpage').innerHTML;
            window.location.href = subpage + '/login';
        }
    }

    function handleBEDBRetrieve(arrayBuffer) { 
        console.log('recv DB retrieve.');
        let message = Protobuf.decode(socketClient, 'MsgDcmInfoCollection', arrayBuffer);
        if (!message) {
            reutrn;
        }

        let dcminfo = message.dcminfo;
        if (!dcminfo || dcminfo.length == 0) {
            return;
        }

        let tbody = document.getElementById('worklist-db');
        tbody.innerHTML = '';
        
        dcminfo.forEach(ele => {
            let tr = '<tr>';
            tr += `<td>${ele.patientName}</td>`;
            tr += `<td>${ele.patientId}</td>`;
            tr += `<td>${ele.seriesId}</td>`;
            tr += `<td>${ele.modality}</td>`;
            tr += '</tr>';   
            tbody.innerHTML += tr; 
        });

        //style changed when choose tr (based on bootstrap)
        $('#table-db tbody tr').click(function() {
            if ($(this).hasClass('success')) {
                $(this).removeClass('success');
            } else {
                $(this).addClass('success').siblings().removeClass('success');
            }
        });
    }

    function handleBEPACSRetrieve(arrayBuffer) {
        let message = Protobuf.decode(socketClient, 'MsgDcmInfoCollection', arrayBuffer);
        if (!message) {
            reutrn;
        }

        let dcminfo = message.dcminfo;
        if (!dcminfo || dcminfo.length == 0) {
            console.log(`can't get dcminfo.`);
            return;
        }

        let tbody = document.getElementById('worklist-pacs');
        tbody.innerHTML = '';
        
        dcminfo.forEach(ele => {
            let tr = '<tr>';
            tr += `<td>${ele.patientName}</td>`;
            tr += `<td>${ele.patientId}</td>`;
            tr += `<td>${ele.seriesId}</td>`;
            tr += `<td>${ele.modality}</td>`;
            tr += '</tr>';   
            tbody.innerHTML += tr; 
        });

        //style changed when choose tr (based on bootstrap)
        $('#table-pacs tbody tr').click(function() {
            if ($(this).hasClass('success')) {
                $(this).removeClass('success');
            } else {
                $(this).addClass('success');
            }
        });
    }

    function annoListDelete(id) {
        let trDelete = $(`#annotation-list #${id}`);
        if (trDelete.length != 0) {
            annotationList.removeChild(trDelete[0]);
        }
    }

    function annoListAdd(id,cx,cy,cz,diameter,probability) {
        if (document.getElementById(id)) {
            console.log('add repeated annotation');
            return;
        }

        let trAdd = document.createElement('tr');
        trAdd.id = id;
        let tdValue = `<td>${cx.toFixed(2) + ',' + cy.toFixed(2) + ',' + cz.toFixed(2)}</td>`;
        tdValue += `<td>${diameter.toFixed(2)}</td>`;
        tdValue += `<td>${probability.toFixed(2)}</td>`;
        trAdd.innerHTML = tdValue;
        annotationList.appendChild(trAdd);

        $(trAdd).click(function(event) {
            $(this).addClass('success').siblings().removeClass('success');
            //send focus annotation message
            let anno_id = $(this).attr('id');
            if (anno_id) {
                sendAnnotationMSG(0, 0, anno_id, ANNOTATION_FOCUS, true, 0, 0, 0, 1.0, socketClient); 
            }
        });
    }

    function annoListModify(id,cx,cy,cz,diameter,probability) {
        let trModify = $(`#annotation-list #${id}`);
        if (trModify.length != 0) {
            trModify[0].innerHTML = '';
            let tdValue= `<td>${cx.toFixed(2) + ',' + cy.toFixed(2) + ',' + cz.toFixed(2)}</td>`;
            tdValue += `<td>${diameter.toFixed(2)}</td>`;
            tdValue += `<td>${probability.toFixed(2)}</td>`;
            trModify[0].innerHTML = tdValue;
        }
    }

    function annoListClean() {
        let annoTableRows = annotationList.rows;
        while (annoTableRows.length > 1) {
            annotationList.deleteRow(annoTableRows.length-1);
        }
    }

    
    function handleEvaluationResult(arrayBuffer) {
        let annotationList = Protobuf.decode(socketClient, 'MsgNoneImgAnnotations', arrayBuffer);
        if (annotationList) {
            let listItems = annotationList.annotation;
            for (let i = 0; i < listItems.length; ++i) {
                let id = listItems[i].id;
                let info = listItems[i].info;
                let status = listItems[i].status;
                let cx = listItems[i].para0;
                let cy = listItems[i].para1;
                let cz = listItems[i].para2;
                let diameter = listItems[i].para3;
                let probability = listItems[i].probability;

                if(status == 0) {//add
                    annoListAdd(id, cx, cy, cz, diameter, probability);
                } else if (status == 1) {// delete
                    annoListDelete(id);
                } else if (status == 2) {// modifying
                    annoListModify(id, cx, cy, cz, diameter, probability);
                }
            }
        }            
    }

    function adjustEvaluationProbabilityThreshold(probability) {
        let buffer = Protobuf.encode(socketClient, 'MsgFloat', {value:probability});
        if (buffer) {
            socketClient.sendData(COMMAND_ID_BE_FE_OPERATION, 
                OPERATION_ID_BE_FE_ADJUST_EVALUATION_PROBABILITY_THRESHOLD, 0, buffer.byteLength, buffer);
        }
    }
    
    function handleBEReady(arrayBuffer) {
        if (!socketClient.protocRoot) {
            console.log('null protocbuf.');
            return;
        }

        revcBEReady = true;

        let msg = Protobuf.decode(socketClient, 'MsgFloat', arrayBuffer);
        if (msg) {
            let probabilityThreshol = msg.value;
            let evaluationProbabilityRange = document.getElementById('range-probability');
            if (evaluationProbabilityRange) {
                evaluationProbabilityRange.onchange = null;
                evaluationProbabilityRange.min = 0;
                evaluationProbabilityRange.max = 1;
                evaluationProbabilityRange.step = 0.025;
                evaluationProbabilityRange.defaultValue = probabilityThreshol;
                evaluationProbabilityRange.value = probabilityThreshol;
                evaluationProbabilityRange.onchange = function() {
                    adjustEvaluationProbabilityThreshold(this.value);
                };
            }
        }        
    }

    //FE command recv package context(binary stream)
    let packageBuffer;
    function recvPackageBuffer(buffer, bufferOffset, dataLen, restDataLen, withHeader) {
        if (withHeader) {
            packageBuffer = new ArrayBuffer(dataLen + restDataLen);
        }
        let dstview = new Uint8Array(packageBuffer);
        let srcview = new Uint8Array(buffer, bufferOffset, dataLen);
        let cpSrc = packageBuffer.byteLength - (dataLen + restDataLen);
        for (let i = 0; i < dataLen; i++) {
            dstview[i+cpSrc] = srcview[i];
        }
        return restDataLen <= 0; 
    }
    function cmdHandler(cmdID, cellID, opID, buffer, bufferOffset, dataLen, restDataLen, withHeader) {
        switch (cmdID) {
            case COMMAND_ID_FE_BE_SEND_IMAGE:
                cells[cellID].handleJpegBuffer(buffer, bufferOffset, dataLen, restDataLen, withHeader);
                break;
            case COMMAND_ID_FE_BE_SEND_NONE_IMAGE:
                if (recvPackageBuffer(buffer, bufferOffset, dataLen, restDataLen, withHeader))  {
                    cells[cellID].handleNongImgBuffer(packageBuffer);
                }
                break;
            case COMMAND_ID_FE_BE_READY:
                if (recvPackageBuffer(buffer, bufferOffset, dataLen, restDataLen, withHeader))  {
                    handleBEReady(packageBuffer);
                }
                break;
            case COMMAND_ID_FE_BE_DB_RETRIEVE_RESULT:
                if (recvPackageBuffer(buffer, bufferOffset, dataLen, restDataLen, withHeader))  {
                    handleBEDBRetrieve(packageBuffer);
                }
                break;
            case COMMAND_ID_FE_BE_HEARTBEAT:
                keepHeartbeat();
                break;
            case COMMAND_ID_FE_BE_SEND_ANNOTATION_LIST:
                if (recvPackageBuffer(buffer, bufferOffset, dataLen, restDataLen, withHeader))  {
                    handleEvaluationResult(packageBuffer);
                }
                break;
            case COMMAND_ID_FE_BE_PACS_RETRIEVE_RESULT:
                if (recvPackageBuffer(buffer, bufferOffset, dataLen, restDataLen, withHeader))  {
                    handleBEPACSRetrieve(packageBuffer);
                }
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
        socketClient.sendData(COMMAND_ID_BE_FE_OPERATION, OPERATION_ID_BE_FE_RESIZE, 0, msgBuffer.byteLength, msgBuffer);
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
        socketClient.sendData(COMMAND_ID_BE_FE_OPERATION, OPERATION_ID_BE_FE_RESIZE, 0, msgBuffer.byteLength, msgBuffer);
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
        socketClient.sendData(COMMAND_ID_BE_FE_OPERATION, OPERATION_ID_BE_FE_INIT, 0, msgBuffer.byteLength, msgBuffer);
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
        socketClient.sendData(COMMAND_ID_BE_FE_DB_RETRIEVE, 0, 0, null);
    }

    function retrievePACS() {
        if (!revcBEReady) {
            console.log('BE not ready!');
            alert('BE not ready!');
            return;
        }
        socketClient.sendData(COMMAND_ID_BE_FE_PACS_RETRIEVE, 0, 0, null);
    }

    function fetchPACS() {
        if (!revcBEReady) {
            console.log('BE not ready!');
            alert('BE not ready!');
            return;
        }

        let choosed = $('#table-pacs tbody tr');
        let choosedIndex = '';
        Array.from(choosed).forEach((item,index)=>{
            if($(item).hasClass('success')) {
                choosedIndex += index + '|';
            }
        });
        console.log(choosedIndex);

        if (!choosedIndex) {
            alert('please choose one series to fetch.');
            reutrn;
        }

        let buffer = Protobuf.encode(socketClient, 'MsgString', {context:choosedIndex});
        if (buffer) {
            socketClient.sendData(COMMAND_ID_BE_FE_PACS_FETCH, 0, 0, buffer.byteLength, buffer);
        }        
    }

    function playVR() {
        socketClient.sendData(COMMAND_ID_BE_FE_PLAY_VR, 0, 3, null);
    }

    function prepare() {
        let username = document.getElementById('username').innerHTML;
        login();

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
                let series = $('#table-db tbody tr.success td:nth-child(3)').html();
                if (!series) {
                    alert('please choose one series.');
                    reutrn;
                }
                document.getElementById('worklist-db-div').hidden = true;
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

        annotationList = document.getElementById("annotation-list");

        let deleteAnnotationBtn = document.getElementById('btn-delete-annotation');
        if (deleteAnnotationBtn) {
            deleteAnnotationBtn.onclick = function(event) {
                let choosedItem = $('#annotation-list tr.success');
                if (choosedItem.length > 0) {
                    let id = choosedItem.attr('id')
                    if (id) {
                        sendAnnotationMSG(0, 0, id, ANNOTATION_DELETE, false, 0, 0, 0, 0, socketClient);//Delete msg
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

            let buffer = Protobuf.encode(socketClient, 'MsgMPRMaskOverlay', {flag:flag, opacity:opacity});
            if (buffer) {
                socketClient.sendData(COMMAND_ID_BE_FE_OPERATION, OPERATION_ID_BE_FE_MPR_MASK_OVERLAY, 0, buffer.byteLength, buffer);
            }
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
                document.getElementById('worklist-db-div').hidden = false;
                document.getElementById('review-div').hidden = true;
                //send back worklist to BE
                socketClient.sendData(COMMAND_ID_BE_FE_BACK_TO_WORKLIST, 0, 0, null);
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
            let buffer = Protobuf.encode(socketClient, 'MsgString', {context:obj.innerHTML});
            if (buffer) {
                socketClient.sendData(COMMAND_ID_BE_FE_OPERATION, OPERATION_ID_BE_FE_SWITCH_PRESET_WINDOWING, 0, buffer.byteLength, buffer);
            }
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
            let buffer = Protobuf.encode(socketClient, 'MsgString', {context:context});
            if (buffer) {
                socketClient.sendData(COMMAND_ID_BE_FE_OPERATION, OPERATION_ID_BE_FE_SWITCH_PRESET_VRT, 0, buffer.byteLength, buffer);
            }
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

        function backToLogin() {
            let subpage = document.getElementById('subpage').innerHTML;
            window.location.href = subpage+'/login';
        }

        let loginBtn = document.getElementById('btn-login-0');
        if (loginBtn) {
            loginBtn.onclick = function() {
                backToLogin();
            }
        }
        loginBtn = document.getElementById('btn-login-1');
        if (loginBtn) {
            loginBtn.onclick = function() {
                backToLogin();
            }
        }

        loginBtn = document.getElementById('btn-login-2');
        if (loginBtn) {
            loginBtn.onclick = function() {
                backToLogin();
            }
        }

        function rollDBPACS() {
            if (document.getElementById('worklist-pacs-div').hidden) {
                document.getElementById('worklist-pacs-div').hidden = false;
                document.getElementById('worklist-db-div').hidden = true;
                document.getElementById('review-div').hidden = true;
            } else {
                document.getElementById('worklist-pacs-div').hidden = true;
                document.getElementById('worklist-db-div').hidden = false;
                document.getElementById('review-div').hidden = true;
            }
        }

        let pacsBtn = document.getElementById('btn-pacs-0');
        if (pacsBtn) {
            pacsBtn.onclick = function() {
                rollDBPACS();
            }
        }
        pacsBtn = document.getElementById('btn-pacs-1');
        if (pacsBtn) {
            pacsBtn.onclick = function() {
                rollDBPACS();
            }
        }

        let queryPACSBtn = document.getElementById('btn-retrieve-pacs');
        if (queryPACSBtn) {
            queryPACSBtn.onclick = function() {
                retrievePACS();
            }
        }

        let fetchPACSBtn = document.getElementById('btn-fetch-pacs');
        if (fetchPACSBtn) {
            fetchPACSBtn.onclick = function() {
                fetchPACS();
            }
        }

        let anonymousBtn = document.getElementById('btn-anonymous');
        if (anonymousBtn) {
            anonymousBtn.onclick = function() {
                let glyphIcon = $('#btn-anonymous span');
                if(glyphIcon.hasClass('glyphicon-eye-open')) {
                    glyphIcon.removeClass('glyphicon-eye-open');
                    glyphIcon.addClass('glyphicon-eye-close');
                    let buffer = Protobuf.encode(socketClient, 'MsgFlag', {flag:true});
                    if (buffer) {
                        socketClient.sendData(COMMAND_ID_BE_FE_ANONYMIZATION, 0, 0, buffer.byteLength, buffer);
                    }
                } else {
                    glyphIcon.removeClass('glyphicon-eye-close');
                    glyphIcon.addClass('glyphicon-eye-open');
                    let buffer = Protobuf.encode(socketClient, 'MsgFlag', {flag:false});
                    if (buffer) {
                        socketClient.sendData(COMMAND_ID_BE_FE_ANONYMIZATION, 0, 0, buffer.byteLength, buffer);
                    }
                }
            }
        }

        // register window quit linsener
        window.onbeforeunload = function(event) {
            logout();
        }
        window.onresize = function() {
            resize()
        };

        //trigger on heartbeat
        let checkHeartbearFunc = setInterval(function() {
            checkHeartbeat();
        }, HEARTBEAT_INTERVAL);
    }

    prepare();
})();