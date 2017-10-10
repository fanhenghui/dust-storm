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
var annotationTBody = null;

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
        for (var i = cpSrc; i < dataLen; i++) {
            dstview[i] = srcview[i];
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
        for (var i = cpSrc; i < dataLen; i++) {
            dstview[i] = srcview[i];
        }

        if (restDataLen <= 0) {
            console.log('recv annotation list.');
            var MsgAnnotationListType = socketClient.protocRoot.lookup('medical_imaging.MsgAnnotationList');
            if (!MsgAnnotationListType) {
                console.log('get annotation list message type failed.');
            }
            var annotationListView = new Uint8Array(annotationListBuffer);
            var annotationList = MsgAnnotationListType.decode(annotationListView);
            if (annotationList) {
                var listItems = annotationList.item;
                for (var i = 0; i < listItems.length; ++i) {
                    var id = listItems[i].id;
                    var info = listItems[i].info;
                    var row = listItems[i].row;
                    var status = listItems[i].status;
                    var cx = listItems[i].para0;
                    var cy = listItems[i].para1;
                    var cz = listItems[i].para2;
                    var diameter = listItems[i].para3;

                    if(status == 0) {//add
                        var rowItem = document.createElement('tr');
                        $(rowItem).click(function(event) {
                            $(this).addClass('success').siblings().removeClass('success');
                        });

                        rowItem.setAttribute('id', id);
                        var cellItem0 = document.createElement('td');
                        cellItem0.innerHTML = cx.toFixed(2) + ',' + cy.toFixed(2) + ',' + cz.toFixed(2);
                        var cellItem1 = document.createElement('td');
                        cellItem1.innerHTML = diameter.toFixed(2);  
                        var cellItem2 = document.createElement('td');
                        cellItem2.innerHTML = info;
                        rowItem.appendChild(cellItem0);
                        rowItem.appendChild(cellItem1);
                        rowItem.appendChild(cellItem2);
                        annotationTBody.appendChild(rowItem);

                    } else if (status == 1) {// delete
                        var childNodes = annotationTBody.childNodes;
                        if (childNodes.length >= row + 1) {
                            annotationTBody.removeChild(childNodes[row + 1]);
                        }
                    } else if (status == 2) {// modifying
                        var childNodes = annotationTBody.childNodes;
                        if (childNodes.length >= row + 1) {
                            var cellsItem = childNodes[row + 1].cells;
                            cellsItem[0].innerHTML = cx.toFixed(2) + ',' + cy.toFixed(2) + ',' + cz.toFixed(2);
                            cellsItem[1].innerHTML = diameter.toFixed(2);  
                            cellsItem[2].innerHTML = info;
                        }
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
        var w = (cellContainerW - 66) / 2;
        var h = (window.innerHeight - navigatorHeight - 60) / 2;
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
        }

        //init default cell action
        cells[0].activeAction(ACTION_ID_MPR_PAGING);
        cells[1].activeAction(ACTION_ID_MPR_PAGING);
        cells[2].activeAction(ACTION_ID_MPR_PAGING);
        cells[3].activeAction(ACTION_ID_ROTATE);

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
                cells[0].activeAction(ACTION_ID_MPR_PAGING);
                cells[1].activeAction(ACTION_ID_MPR_PAGING);
                cells[2].activeAction(ACTION_ID_MPR_PAGING);
                cells[3].activeAction(ACTION_ID_ROTATE);
                break;
            case 'common-tool-zoom':
                for (var i = 0; i < cells.length; ++i) {
                    cells[i].activeAction(ACTION_ID_ZOOM);
                }
                break;
            case 'common-tool-pan':
                for (var i = 0; i < cells.length; ++i) {
                    cells[i].activeAction(ACTION_ID_PAN);
                }
                break;
            case 'common-tool-rotate':
                for (var i = 0; i < cells.length; ++i) {
                    cells[i].activeAction(ACTION_ID_ROTATE);
                }
                break;
            case 'common-tool-windowing':
                for (var i = 0; i < cells.length; ++i) {
                    cells[i].activeAction(ACTION_ID_WINDOWING);
                }
                break;
            case 'common-tool-annotation':
                cells[0].activeAction(ACTION_ID_MRP_ANNOTATION);
                cells[1].activeAction(ACTION_ID_MRP_ANNOTATION);
                cells[2].activeAction(ACTION_ID_MRP_ANNOTATION);
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
            searchWorklistBtn.addEventListener('click', function(event) {
                searchWorklist();
            });
        } else {
            console.log('get searchBtn node failed.');
        }
        
        var loadSeriesBtn = document.getElementById('btn-load-series');
        if (loadSeriesBtn) {
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
        } else {
            console.log('get loadBtn node failed.');
        }

        var comToolsDiv = document.getElementById('common-tools');
        if (comToolsDiv) {
            var comToolsBtns = comToolsDiv.getElementsByTagName('button');
            for (var i = 0; i < comToolsBtns.length; ++i) {
                comToolsBtns[i].addEventListener('click', function(event) {
                    switchCommonTool(this.id);
                });
            }
        } else {
            console.log('get common-tools failed.');
        }

        var playVRBtn = document.getElementById('btn-play-vr');
        if (playVRBtn) {
            playVRBtn.addEventListener('click', function() {
                playVR();
            });
        } else {
            console.log('get btn-play-vr node failed.');
        }

        annotationTBody = document.getElementById("annotation-list");

        var deleteAnnotationBtn = document.getElementById('btn-delete-annotation');
        if (deleteAnnotationBtn) {
            deleteAnnotationBtn.addEventListener('click', function(event) {
                var choosedItem = $('#annotation-list tr.success');
                if (choosedItem) {
                    var id = choosedItem.attr('id')
                    sendMSG(0, 0, id, 1, false, 0, 0, 0, socketClient);//Delete msg
                }

            });
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