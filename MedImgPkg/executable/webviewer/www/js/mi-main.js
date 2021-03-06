
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

    //pacs result
    let pacs_study_number = 0;
    let pacs_page_sum = 0;
    let pacs_current_page = 1;

    //db result
    let db_study_number = 0;
    let db_page_sum = 0;
    let db_current_page = 1;

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
            socket.userName = document.getElementById('username').innerHTML;
            socket.userID = getUserID(socket.userName);
            const onlineToken = $.cookie('online_token');
            
            socketClient = new SocketClient(socket);
            socketClient.loadProtoc(PROTOBUF_BE_FE);
            socket.emit('login', {
                userID: socket.userID,
                userName: socket.userName,
                onlineToken: onlineToken,
            });
            socket.on('data', tcpBuffer => {
                socketClient.recvData(tcpBuffer, cmdHandler);
            });
            socket.on('login_out', function() {
                loginOut();
            });
        }
    }

    function loginOut() {
        window.location.href = '/user/logout';
    }

    let heartbeatCount = 1;
    let lastheartbearCount = 0;
    const HEARTBEAT_INTERVAL = 6 * 1000; //6s
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
            window.location.href = '/user/login';
        }
    }
 
    function handleBEDBQuery(arrayBuffer) { 
        let message = Protobuf.decode(socketClient, 'MsgStudyWrapperCollection', arrayBuffer);
        if (!message) {
            return;
        }

        let studyWrappers = message.studyWrappers;
        if (!studyWrappers || studyWrappers.length == 0) {
            $('#db-study-info').html('');
            $('#worklist-db-study').html('');
            $('#worklist-db-series').html('');
            $('#btn-db-page-up').addClass('disabled');
            $('#btn-db-page-down').addClass('disabled');
            return;
        }

        db_study_number = message.numStudy <=0 ? 0 : message.numStudy;
        db_page_sum = Math.ceil(db_study_number/LIST_CAPACITY);
        db_current_page = 1;
        let study_from = (db_current_page-1)*LIST_CAPACITY + 1;
        let study_to = db_current_page*LIST_CAPACITY;
        study_to = study_to > db_study_number ? db_study_number : study_to;
        $('#db-study-info').html(`Study ${study_from} to ${study_to} of ${db_study_number}`);
        if (db_page_sum > 1) {
            $('#btn-db-page-down').removeClass('disabled');
        }
        let tbody = document.getElementById('worklist-db-study');
        tbody.innerHTML = '';
        
        studyWrappers.forEach(ele => {
            let tr = `<tr studyidx="${ele.studyInfo.id}">`;
            tr += `<td>${ele.patientInfo.patientName}</td>`;
            tr += `<td>${ele.patientInfo.patientId}</td>`;
            tr += `<td>${ele.patientInfo.patientBirthDate}</td>`;
            tr += `<td>${ele.patientInfo.patientSex}</td>`;
            tr += `<td>${ele.studyInfo.studyDate}</td>`;
            tr += `<td>${ele.studyInfo.studyDesc}</td>`;
            tr += `<td>${ele.studyInfo.numSeries}</td>`;
            tr += '</tr>';   
            tbody.innerHTML += tr; 
        });

        //style changed when choose tr (based on bootstrap)
        $('#table-db-study tbody tr').click(function() {
            $(this).addClass('success').siblings().removeClass('success');

            let buffer = Protobuf.encode(socketClient, 'MsgInt', {value:this.getAttribute('studyidx')});
            if (buffer) {
                socketClient.sendData(COMMAND_ID_BE_FE_DB_GET_SERIES_LIST, 0, 0, buffer.byteLength, buffer);
            }
        });

        $('#worklist-db-series').html("");
    }

    function handleBEDBSeriesList(arrayBuffer) {
        let message = Protobuf.decode(socketClient, 'MsgStudyWrapper', arrayBuffer);
        if (!message) {
            return;
        }

        let seriesInfos = message.seriesInfos;
        if (!seriesInfos || seriesInfos.length == 0) {
            console.log(`can't get series infos.`);
            return;
        }

        let tbody = document.getElementById('worklist-db-series');
        tbody.innerHTML = '';
        
        seriesInfos.forEach(ele => {
            let tr = `<tr seriesidx="${ele.id}">`;
            tr += `<td>${ele.seriesNo}</td>`;
            tr += `<td>${ele.modality}</td>`;
            tr += `<td>${ele.seriesDesc}</td>`;
            tr += `<td>${ele.numInstance}</td>`;
            tr += '</tr>';
            tbody.innerHTML += tr; 
        });

        $('#table-db-series tbody tr').click(function() {
            $(this).addClass('success').siblings().removeClass('success');
        });
    }

    function handleBEPACSQuery(arrayBuffer) {
        let message = Protobuf.decode(socketClient, 'MsgStudyWrapperCollection', arrayBuffer);
        if (!message) {
            return;
        }

        let studyWrappers = message.studyWrappers;
        if (!studyWrappers || studyWrappers.length == 0) {
            $('#pacs-study-info').html('');
            $('#worklist-pacs-study').html('');
            $('#worklist-pacs-series').html('');
            $('#btn-pacs-page-up').addClass('disabled');
            $('#btn-pacs-page-down').addClass('disabled');
            return;
        }

        pacs_study_number = message.numStudy <=0 ? 0 : message.numStudy;
        pacs_page_sum = Math.ceil(pacs_study_number/LIST_CAPACITY);
        pacs_current_page = 1;
        let study_from = (pacs_current_page-1)*LIST_CAPACITY + 1;
        let study_to = pacs_current_page*LIST_CAPACITY;
        study_to = study_to > pacs_study_number ? pacs_study_number : study_to;
        $('#pacs-study-info').html(`Study ${study_from} to ${study_to} of ${pacs_study_number}`);
        if (pacs_page_sum > 1) {
            $('#btn-pacs-page-down').removeClass('disabled');
        }
        let tbody = document.getElementById('worklist-pacs-study');
        tbody.innerHTML = '';
        
        studyWrappers.forEach(ele => {
            let tr = `<tr studyidx="${ele.studyInfo.id}">`;
            tr += `<td>${ele.patientInfo.patientName}</td>`;
            tr += `<td>${ele.patientInfo.patientId}</td>`;
            tr += `<td>${ele.patientInfo.patientBirthDate}</td>`;
            tr += `<td>${ele.patientInfo.patientSex}</td>`;
            tr += `<td>${ele.studyInfo.studyDate}</td>`;
            tr += `<td>${ele.studyInfo.studyDesc}</td>`;
            tr += `<td>${ele.studyInfo.numSeries}</td>`;
            tr += '</tr>';   
            tbody.innerHTML += tr; 
        });

        //style changed when choose tr (based on bootstrap)
        $('#table-pacs-study tbody tr').click(function() {
            // if ($(this).hasClass('success')) {
            //     $(this).removeClass('success');
            // } else {
            //     $(this).addClass('success');
            // }
            $(this).addClass('success').siblings().removeClass('success');

            let buffer = Protobuf.encode(socketClient, 'MsgInt', {value:this.getAttribute('studyidx')});
            if (buffer) {
                socketClient.sendData(COMMAND_ID_BE_FE_PACS_GET_SERIES_LIST, 0, 0, buffer.byteLength, buffer);
            }
        });
        $('#worklist-pacs-series').html("");
    }

    function handleBEPACSStudyList(arrayBuffer) {
        let message = Protobuf.decode(socketClient, 'MsgStudyWrapperCollection', arrayBuffer);
        if (!message) {
            return;
        }

        let studyWrappers = message.studyWrappers;
        if (!studyWrappers || studyWrappers.length == 0) {
            console.log(`can't get study wrappers.`);
            return;
        }

        let tbody = document.getElementById('worklist-pacs-study');
        tbody.innerHTML = '';
        
        studyWrappers.forEach(ele => {
            let tr = `<tr studyidx="${ele.studyInfo.id}">`;
            tr += `<td>${ele.patientInfo.patientName}</td>`;
            tr += `<td>${ele.patientInfo.patientId}</td>`;
            tr += `<td>${ele.patientInfo.patientBirthDate}</td>`;
            tr += `<td>${ele.patientInfo.patientSex}</td>`;
            tr += `<td>${ele.studyInfo.studyDate}</td>`;
            tr += `<td>${ele.studyInfo.studyDesc}</td>`;
            tr += `<td>${ele.studyInfo.numSeries}</td>`;
            tr += '</tr>';   
            tbody.innerHTML += tr; 
        });

        //style changed when choose tr (based on bootstrap)
        $('#table-pacs-study tbody tr').click(function() {
            // if ($(this).hasClass('success')) {
            //     $(this).removeClass('success');
            // } else {
            //     $(this).addClass('success');
            // }
            $(this).addClass('success').siblings().removeClass('success');

            let buffer = Protobuf.encode(socketClient, 'MsgInt', {value:this.getAttribute('studyidx')});
            if (buffer) {
                socketClient.sendData(COMMAND_ID_BE_FE_PACS_GET_SERIES_LIST, 0, 0, buffer.byteLength, buffer);
            }
        });

        $('#worklist-pacs-series').html("");
    }

    function handleBEPACSSeriesList(arrayBuffer) {
        let message = Protobuf.decode(socketClient, 'MsgStudyWrapper', arrayBuffer);
        if (!message) {
            return;
        }

        let seriesInfos = message.seriesInfos;
        if (!seriesInfos || seriesInfos.length == 0) {
            console.log(`can't get series infos.`);
            return;
        }

        let tbody = document.getElementById('worklist-pacs-series');
        tbody.innerHTML = '';
        
        seriesInfos.forEach(ele => {
            let tr = `<tr seriesidx="${ele.id}">`;
            tr += `<td>${ele.seriesNo}</td>`;
            tr += `<td>${ele.modality}</td>`;
            tr += `<td>${ele.seriesDesc}</td>`;
            tr += `<td>${ele.numInstance}</td>`;
            tr += '</tr>';
            tbody.innerHTML += tr; 
        });

        $('#table-pacs-series tbody tr').click(function() {
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
            case COMMAND_ID_FE_BE_DB_QUERY_RESULT:
                if (recvPackageBuffer(buffer, bufferOffset, dataLen, restDataLen, withHeader))  {
                    handleBEDBQuery(packageBuffer);
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
            case COMMAND_ID_FE_BE_PACS_QUERY_RESULT:
                if (recvPackageBuffer(buffer, bufferOffset, dataLen, restDataLen, withHeader))  {
                    handleBEPACSQuery(packageBuffer);
                }
                break;
            case COMMAND_ID_FE_BE_PACS_STUDY_LIST_RESULT:
                if (recvPackageBuffer(buffer, bufferOffset, dataLen, restDataLen, withHeader))  {
                    handleBEPACSStudyList(packageBuffer);
                }
                break;
            case COMMAND_ID_FE_BE_PACS_SERIES_LIST_RESULT:
                if (recvPackageBuffer(buffer, bufferOffset, dataLen, restDataLen, withHeader))  {
                    handleBEPACSSeriesList(packageBuffer);
                }
                break;
            case COMMAND_ID_FE_BE_DB_SERIES_LIST_RESULT:
                if (recvPackageBuffer(buffer, bufferOffset, dataLen, restDataLen, withHeader))  {
                    handleBEDBSeriesList(packageBuffer);
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

    function loadSeries() {
        if (!socketClient) {
            console.log('socket client is null.');
            return;
        }
        if (!socketClient.protocRoot) {
            console.log('null protocbuf.')
            return;
        }
        
        //get study&series id
        let chooseStudy = $('#table-db-study tbody tr.success');
        if (!chooseStudy) {
            alert('please choose a study.');
            return;
        }
        let studyIdx = chooseStudy.attr('studyidx');

        let chooseSeries = $('#table-db-series tbody tr.success');
        if (!chooseSeries) {
            alert('please choose a series.');
            return;
        }
        let seriesIdx = chooseSeries.attr('seriesidx');
        seriesUID = seriesIdx;

        let instance_num = $('#table-db-series tbody tr.success td:nth-child(4)').html();
        if (parseInt(instance_num) < 16) {
            alert('series instance number must be large than 16.');
            return;
        }

        document.getElementById('worklist-db-div').hidden = true;
        document.getElementById('review-div').hidden = false;

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

        msgInit.studyPk = studyIdx;
        msgInit.seriesPk = seriesIdx;
        //TODO set user ID to annotation
        msgInit.userID = "";
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

    function queryDB(study_from, study_to) {
        if (!revcBEReady) {
            console.log('BE not ready!');
            alert('BE not ready!');
            return;
        }
        //construct query key
        let queryKey = {
            patientId: $('#db-key-patient-id').val(),
            patientName: $('#db-key-patient-name').val(),
            patientBirthDate: '',
            studyDate: '',
            studyTime: '',
            accessionNo: '',
            modality: $('#db-key-modality').attr('modality'),
            studyFrom: study_from,
            studyTo: study_to,
        };
        let patientBirthDateFrom = $('#db-key-patient-birth-date-from').val().replace(/\-+/g, '');;
        let patientBirthDateTo = $('#db-key-patient-birth-date-to').val().replace(/\-+/g, '');;
        if (patientBirthDateFrom.length !=0 && patientBirthDateTo.length != 0) {
            queryKey.patientBirthDate = `${patientBirthDateFrom}-${patientBirthDateTo}`;
        }

        let studyDateFrom = $('#db-key-study-date-from').val().replace(/\-+/g, '');;
        let studyDateTo = $('#db-key-study-date-to').val().replace(/\-+/g, '');;
        if (studyDateFrom.length !=0 && studyDateTo.length != 0) {
            queryKey.studyDate = `${studyDateFrom}-${studyDateTo}`;
        }

        let buffer = Protobuf.encode(socketClient, 'MsgDcmQueryKey', queryKey);
        if (buffer) {
            socketClient.sendData(COMMAND_ID_BE_FE_DB_QUERY, 0, 0, buffer.byteLength, buffer);
        }
    }

    function queryPACS() {
        if (!revcBEReady) {
            console.log('BE not ready!');
            alert('BE not ready!');
            return;
        }
        //construct query key
        let queryKey = {
            patientId: $('#pacs-key-patient-id').val(),
            patientName: $('#pacs-key-patient-name').val(),
            patientBirthDate: '',
            studyDate: '',
            studyTime: '',
            accessionNo: '',
            modality: $('#pacs-key-modality').attr('modality'),
        };
        let patientBirthDateFrom = $('#pacs-key-patient-birth-date-from').val().replace(/\-+/g, '');;
        let patientBirthDateTo = $('#pacs-key-patient-birth-date-to').val().replace(/\-+/g, '');;
        if (patientBirthDateFrom.length !=0 && patientBirthDateTo.length != 0) {
            queryKey.patientBirthDate = `${patientBirthDateFrom}-${patientBirthDateTo}`;
        }

        let studyDateFrom = $('#pacs-key-study-date-from').val().replace(/\-+/g, '');;
        let studyDateTo = $('#pacs-key-study-date-to').val().replace(/\-+/g, '');;
        if (studyDateFrom.length !=0 && studyDateTo.length != 0) {
            queryKey.studyDate = `${studyDateFrom}-${studyDateTo}`;
        }

        let buffer = Protobuf.encode(socketClient, 'MsgDcmQueryKey', queryKey);
        if (buffer) {
            socketClient.sendData(COMMAND_ID_BE_FE_PACS_QUERY, 0, 0, buffer.byteLength, buffer);
        }
    }

    function retrievePACS() {
        if (!revcBEReady) {
            console.log('BE not ready!');
            alert('BE not ready!');
            return;
        }
        //这里暂时只支持一个study的多个series的retrieve

        let studyIdx = -1;
        let studyList = $('#table-pacs-study tbody tr');
        for (let i = 0; i < studyList.length; ++i) {
            if ($(studyList[i]).hasClass('success')) {
                studyIdx = $(studyList[i]).attr('studyidx');
                break;
            }
        }
        if (studyIdx == -1) {
            alert('please choose a study.');
            return;
        }

        let choosed = $('#table-pacs-series tbody tr');
        let context = {seriesUid:[], studyUid:[]};
        Array.from(choosed).forEach((item,index)=>{
            if($(item).hasClass('success')) {
                context.studyUid.push(studyIdx);
                context.seriesUid.push($(item).attr('seriesidx'));
            }
        });

        let buffer = Protobuf.encode(socketClient, 'MsgDcmPACSRetrieveKey', context);
        if (buffer) {
            socketClient.sendData(COMMAND_ID_BE_FE_PACS_RETRIEVE, 0, 0, buffer.byteLength, buffer);
        }
    }

    function playVR() {
        socketClient.sendData(COMMAND_ID_BE_FE_PLAY_VR, 0, 3, null);
    }

    function prepare() {
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

        let loginBtn = document.getElementById('btn-login-0');
        if (loginBtn) {
            loginBtn.onclick = function() {
                loginOut();
            }
        }
        loginBtn = document.getElementById('btn-login-1');
        if (loginBtn) {
            loginBtn.onclick = function() {
                loginOut();
            }
        }

        loginBtn = document.getElementById('btn-login-2');
        if (loginBtn) {
            loginBtn.onclick = function() {
                loginOut();
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

        //------------------------------------------//
        //query&load from DB
        //------------------------------------------//
        let queryDBBtn = document.getElementById('btn-query-db');
        if (queryDBBtn) {
            queryDBBtn.onclick = function(event) {
                queryDB(0,9);
            };
        } else {
            console.log('get searchBtn node failed.');
        }

        let resetDBQueryKey = document.getElementById('btn-reset-db-query-key');
        if (resetDBQueryKey) {
            resetDBQueryKey.onclick = function() {
                $('#db-key-patient-name').val('');
                $('#db-key-patient-id').val('');
                $('#db-key-patient-birth-date-from').val('');
                $('#db-key-patient-birth-date-to').val('');
                $('#db-key-study-date-from').val('');
                $('#db-key-study-date-to').val('');
                $('#db-key-modality').html('All<span class="caret"></span>');
                $('#db-key-modality').attr('modality','');
            }
        }
        $('#db-key-modality-all').click(function() {
            $('#db-key-modality').html('All<span class="caret"></span>');
            $('#db-key-modality').attr('modality','');
        });
        $('#db-key-modality-ct').click(function() {
            $('#db-key-modality').html('CT<span class="caret"></span>');
            $('#db-key-modality').attr('modality','CT');
        });
        $('#db-key-modality-mr').click(function() {
            $('#db-key-modality').html('MR<span class="caret"></span>');
            $('#db-key-modality').attr('modality','MR');
        });
        $('#db-key-modality-rt-struct').click(function() {
            $('#db-key-modality').html('RT_STRUCT<span class="caret"></span>');
            $('#db-key-modality').attr('modality','RT_STRUCT');
        });

        let pageDownDBtudyList = document.getElementById("btn-db-page-down");
        if (pageDownDBtudyList) {
            pageDownDBtudyList.onclick = function() {
                if (db_current_page < db_page_sum) {
                    db_current_page += 1;

                    let study_from = (db_current_page-1)*LIST_CAPACITY + 1;
                    let study_to = db_current_page*LIST_CAPACITY;
                    study_to = study_to > db_study_number ? db_study_number : study_to;
                    $('#db-study-info').html(`Study ${study_from} to ${study_to} of ${db_study_number}`);
                    if (db_current_page > 1) {
                        $('#btn-db-page-up').removeClass('disabled');
                    }

                    queryDB(study_from, study_to);
                }
                
                if (db_current_page >= db_page_sum) {
                    $('#btn-db-page-down').addClass('disabled');
                }
            }
        }

        let pageUpDBStudyList = document.getElementById("btn-db-page-up");
        if (pageUpDBStudyList) {
            pageUpDBStudyList.onclick = function() {
                if (db_current_page > 1) {
                    db_current_page -= 1;

                    let study_from = (db_current_page-1)*LIST_CAPACITY + 1;
                    let study_to = db_current_page*LIST_CAPACITY;
                    study_to = study_to > db_study_number ? db_study_number : study_to;
                    $('#db-study-info').html(`Study ${study_from} to ${study_to} of ${db_study_number}`);
                    if (db_current_page < db_page_sum) {
                        $('#btn-db-page-down').removeClass('disabled');
                    }

                    queryDB(study_from, study_to-1);
                }
                
                if (db_current_page <= 1) {
                    $('#btn-db-page-up').addClass('disabled');
                }
            }
        }
        
        let loadSeriesBtn = document.getElementById('btn-load-series');
        if (loadSeriesBtn) {
            loadSeriesBtn.onclick = function(event) {
                annoListClean();
                loadSeries();
            };
        } else {
            console.log('get loadBtn node failed.');
        }


        //------------------------------------------//
        //query&retrieve from PACS
        //------------------------------------------//
        let queryPACSBtn = document.getElementById('btn-query-pacs');
        if (queryPACSBtn) {
            queryPACSBtn.onclick = function() {
                queryPACS();
            }
        }
        
        let resetPACSQueryKey = document.getElementById('btn-reset-pacs-query-key');
        if (resetPACSQueryKey) {
            resetPACSQueryKey.onclick = function() {
                $('#pacs-key-patient-name').val('');
                $('#pacs-key-patient-id').val('');
                $('#pacs-key-patient-birth-date-from').val('');
                $('#pacs-key-patient-birth-date-to').val('');
                $('#pacs-key-study-date-from').val('');
                $('#pacs-key-study-date-to').val('');
                $('#pacs-key-modality').html('All<span class="caret"></span>');
                $('#pacs-key-modality').attr('modality','');
            }
        }
        $('#pacs-key-modality-all').click(function() {
            $('#pacs-key-modality').html('All<span class="caret"></span>');
            $('#pacs-key-modality').attr('modality','');
        });
        $('#pacs-key-modality-ct').click(function() {
            $('#pacs-key-modality').html('CT<span class="caret"></span>');
            $('#pacs-key-modality').attr('modality','CT');
        });
        $('#pacs-key-modality-mr').click(function() {
            $('#pacs-key-modality').html('MR<span class="caret"></span>');
            $('#pacs-key-modality').attr('modality','MR');
        });
        $('#pacs-key-modality-rt-struct').click(function() {
            $('#pacs-key-modality').html('RT_STRUCT<span class="caret"></span>');
            $('#pacs-key-modality').attr('modality','RT_STRUCT');
        });

        let pageDownPACSStudyList = document.getElementById("btn-pacs-page-down");
        if (pageDownPACSStudyList) {
            pageDownPACSStudyList.onclick = function() {
                if (pacs_current_page < pacs_page_sum) {
                    pacs_current_page += 1;

                    let study_from = (pacs_current_page-1)*LIST_CAPACITY + 1;
                    let study_to = pacs_current_page*LIST_CAPACITY;
                    study_to = study_to > pacs_study_number ? pacs_study_number : study_to;
                    $('#pacs-study-info').html(`Study ${study_from} to ${study_to} of ${pacs_study_number}`);
                    if (pacs_current_page > 1) {
                        $('#btn-pacs-page-up').removeClass('disabled');
                    }

                    let buffer = Protobuf.encode(socketClient, 'MsgListPage', {from:study_from-1, to:study_to});
                    if (buffer) {
                        socketClient.sendData(COMMAND_ID_BE_FE_PACS_GET_STUDY_LIST, 0, 0, buffer.byteLength, buffer);
                    }
                }
                
                if (pacs_current_page >= pacs_page_sum) {
                    $('#btn-pacs-page-down').addClass('disabled');
                }
            }
        }

        let pageUpPACSStudyList = document.getElementById("btn-pacs-page-up");
        if (pageUpPACSStudyList) {
            pageUpPACSStudyList.onclick = function() {
                if (pacs_current_page > 1) {
                    pacs_current_page -= 1;

                    let study_from = (pacs_current_page-1)*LIST_CAPACITY + 1;
                    let study_to = pacs_current_page*LIST_CAPACITY;
                    study_to = study_to > pacs_study_number ? pacs_study_number : study_to;
                    $('#pacs-study-info').html(`Study ${study_from} to ${study_to} of ${pacs_study_number}`);
                    if (pacs_current_page < pacs_page_sum) {
                        $('#btn-pacs-page-down').removeClass('disabled');
                    }

                    let buffer = Protobuf.encode(socketClient, 'MsgListPage', {from:study_from-1, to:study_to});
                    if (buffer) {
                        socketClient.sendData(COMMAND_ID_BE_FE_PACS_GET_STUDY_LIST, 0, 0, buffer.byteLength, buffer);
                    }
                }
                
                if (pacs_current_page <= 1) {
                    $('#btn-pacs-page-up').addClass('disabled');
                }
            }
        }

        let retrievePACSBtn = document.getElementById('btn-retrieve-pacs');
        if (retrievePACSBtn) {
            retrievePACSBtn.onclick = function() {
                retrievePACS();
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
            //Socket disconnect
            if (socket != null) {
                socket.emit('disconnect', {userID: socket.userID,userName: socket.userName});
                location.reload();
            }
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