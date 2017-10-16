//corner info marin
const TEXT_MARGIN = 4;
const DOUBLE_CLICK_INTERVAL = 250;

function Cell(cellName, cellID, canvas, svg, socketClient) {
    this.cellName = cellName;
    this.cellID = cellID;
    
    //containee canvas& svg
    this.canvas = canvas;
    this.svg = svg;

    //web-socket-client
    this.socketClient = socketClient;

    //canvas JPEG based draw
    this.jpegStr = '';
    this.jpegImg = new Image();

    //svg(d3.js) based graphic
    this.noneImgBuf = null;

    //mouse event
    this.mouseBtn = BTN_NONE;
    this.mouseStatus = BTN_UP;
    this.mousePre = {x:0,y:0};
    this.mouseClickTick = 0;

    //mouse action
    this.mouseCurAction = null;
    this.mouseActionCommon = new ActionCommon(socketClient, cellID);
    this.mouseActionAnnotation = new ActionAnnotation(socketClient, cellID);
    this.mouseDoubleClickEvent = null;
    this.mouseFocus = false;
    this.mouseFocusEvent = null;

    //annotation ROI
    this.rois = [];
    this.lastROI = null;

    //crosshair
    this.crosshair = null;
}

function refreshCanvas(canvas, img) {
    canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
}

Cell.prototype.handleJpegBuffer = function(tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader) {
    if (withHeader) {
        this.jpegStr = '';
    } 
    var imgBuffer = new Uint8Array(tcpBuffer, bufferOffset, dataLen);
    this.jpegStr += String.fromCharCode.apply(null, imgBuffer);
    if(restDataLen <= 0) {
        this.jpegImg.src =  'data:image/jpg;base64,' + btoa(this.jpegStr);
        loadImg = (function(){
            refreshCanvas(this.canvas, this.jpegImg);
        }).bind(this);
        this.jpegImg.onload = loadImg;
    }
}

Cell.prototype.handleNongImgBuffer = function (tcpBuffer, bufferOffset, dataLen, restDataLen, withHeader) {
    if (withHeader) {
        noneImgBuf = new ArrayBuffer(dataLen + restDataLen);
    }
    var dstRecord = new Uint8Array(noneImgBuf);
    var srcRecord = new Uint8Array(tcpBuffer, bufferOffset, dataLen);
    var cpSrc = noneImgBuf.byteLength - (dataLen + restDataLen);
    for (var i = cpSrc; i < dataLen; i++) {
        dstRecord[i] = srcRecord[i];
    }

    // sucessfully receive all none-image
    if (restDataLen <= 0) {
        //console.log('receive Nong Img Buffer.');
        var MsgNoneImgCollection = socketClient.protocRoot.lookup('medical_imaging.MsgNoneImgCollection');
        if (!MsgNoneImgCollection) {
            console.log('get MsgNoneImgCollection type failed.');
        }

        //decode the byte array with protobuffer
        var noneImgBufView = new Uint8Array(noneImgBuf);
        var receivedMsg = MsgNoneImgCollection.decode(noneImgBufView);
        if (receivedMsg.cornerInfos) {
            var txt = receivedMsg.cornerInfos.infos;//MSG Format "LT|1:patientName|2:patientID\nLB....\nRT|....\nRB|....\n"
            var corners = txt.split('\n');

            // for each corner
            // 'length-1': avoid the last empty string "" after '\n'
            for (var i = 0; i < (corners.length - 1); ++i) {
                var oneCornerInfo = corners[i].split('|');
                let cornerTag = oneCornerInfo[0];
                var infoContext = [];
                for (var j = 1; j < oneCornerInfo.length; ++j) {
                    var info = oneCornerInfo[j].split(',');
                    infoContext.push({pos:info[0], context:info[1]});
                }
                
                var svgText = d3.select(this.svg).select('#' + cornerTag);
                switch(cornerTag) {
                    case 'LT':
                    var xpos = TEXT_MARGIN;
                    var txtspacing = parseFloat(svgText.attr('font-size'));
                    var texorigin = 1.2*txtspacing;
                    break;
                    case 'RT':
                    var xpos = this.canvas.width - TEXT_MARGIN;
                    var txtspacing = parseFloat(svgText.attr('font-size'));
                    var texorigin = 1.2*txtspacing;
                    break;
                    
                    case 'LB':
                    var xpos = TEXT_MARGIN;
                    var txtspacing = -parseFloat(svgText.attr('font-size'));
                    var texorigin = this.canvas.height + 0.2*txtspacing;
                    break;
                    case 'RB':
                    var xpos = this.canvas.width - TEXT_MARGIN;
                    var txtspacing = -parseFloat(svgText.attr('font-size'));
                    var texorigin = this.canvas.height + 0.2*txtspacing;
                    break;
                }                

                var tspans = svgText.selectAll('tspan');
                if(tspans.empty())
                {
                    tspans.data(infoContext).enter()
                    .append('tspan')
                    .attr('x', xpos)
                    .attr('y', function (d) {
                        return texorigin + (d.pos)*txtspacing;
                    })
                    .attr('id', function (d) {
                        return cornerTag + '-' + d.pos;
                    })
                    .text(function (d, i) {
                        return d.context;
                    });
                } else {
                    tspans.data(infoContext, function(d) {//change data[d.pos]
                        return d.pos;
                    })
                    .text(function (d, i) {
                        return d.context;
                    });
                }
            }
        }
        
        if (receivedMsg.annotations) {
            var annotations = receivedMsg.annotations.annotation;
            if (annotations) {
                for (var i = 0; i < annotations.length; ++i) {
                    var annoUnit = annotations[i];
                    var id = annoUnit.id;
                    var status = annoUnit.status;
                    var vis = annoUnit.visibility;
                    var cx = annoUnit.para0;
                    var cy = annoUnit.para1;
                    var r = annoUnit.para2;
                    switch (annoUnit.status) {
                        case 0: //add
                            this.rois.push(this.mouseActionAnnotation.createROICircle(id, this.svg, cx, cy, r, (vis!=0)));
                            break;
                        case 1: //delete
                            for (var i = 0; i < this.rois.length; ++i) {
                                if (this.rois[i].key == id) {
                                    this.rois[i].release();
                                    this.rois.splice(i, 1);
                                    break;
                                }
                            }
                            break;
                        case 2: //modifying
                            for (var i = 0; i < this.rois.length; ++i) {
                                if (this.rois[i].key == id) {
                                    this.rois[i].locate(cx, cy, r);
                                    this.rois[i].visible(vis)
                                    break;
                                }
                            }
                            break;
                    }                    
                }
            }
        }

        //TODO crosshair message
        if (receivedMsg.crosshair) {
            if (this.crosshair) {
                this.crosshair.parseNoneImg(receivedMsg.crosshair);
            }
        }

    }
}

Cell.prototype.resize = function (width, height) {
    //canvas&svg resize
    //send msg to notigy BE resize will be call in main
    this.canvas.width = Math.floor(width);
    this.canvas.height = Math.floor(height);
    var top = this.canvas.offsetTop;
    var left = this.canvas.offsetLeft;
    var viewBox = left.toString() + ' ' + top + ' ' + Math.floor(width) + ' ' + Math.floor(height);
    this.svg.setAttribute('viewBox', viewBox);
    this.svg.setAttribute('width', Math.floor(width));
    this.svg.setAttribute('height', Math.floor(height));
    this.svg.setAttribute('x', left);
    this.svg.setAttribute('y', top);


    // resize the txts located at 4 corners
    d3.select(this.svg)
        .selectAll('text')
        .attr('x', function (d, i) {
            return (i < 2) ? TEXT_MARGIN : width - TEXT_MARGIN;
        })
        .attr('y', function (d, i) {
            return (i % 2 == 0) ? TEXT_MARGIN : height - TEXT_MARGIN;
        })
        .each(function (d, i) {
            d3.select(this).selectAll('tspan')
                .attr('x', function (datum, j) {
                    return (i < 2) ? TEXT_MARGIN : width - TEXT_MARGIN;
                })
                .attr('y', function (datum, j) {
                    var txtspacing = parseFloat(d3.select('text').attr('font-size'));
                    if (i % 2 == 0) {
                        var texorigin = 1.2 * txtspacing;
                        return (texorigin + datum.pos * txtspacing);
                    } else {
                        var texorigin = height + 0.2 * (-txtspacing);
                        return (texorigin + datum.pos * (-txtspacing));
                    }
                })
        });
    
    for (var i = 0; i < this.rois.length; ++i) {
        this.rois[i].resize(width, height);
    }
}

Cell.prototype.mouseClickTicker = function() {
    if (this.mouseClickTick >= 2) {
        //double click event
        if (this.mouseDoubleClickEvent) {
            this.mouseDoubleClickEvent(this.cellID);
        }
    }
    //reset tick
    this.mouseClickTick = 0;
}

Cell.prototype.mouseDown = function(event) {
    this.mouseStatus = BTN_DOWN;
    this.mouseBtn = event.button;
    this.mouseClickTick += 1;//mouse click tick for check double click
    if (!this.mouseFocus && this.mouseFocusEvent) {
        this.mouseFocusEvent(this.cellID);
    }

    var x = event.clientX - this.svg.getBoundingClientRect().left;
    var y = event.clientY - this.svg.getBoundingClientRect().top;
    this.mousePre.x = x;
    this.mousePre.y = y;

    setTimeout((function(event) {
        this.mouseClickTicker();
    }).bind(this), DOUBLE_CLICK_INTERVAL);

    this.mouseCurAction.mouseDown(this.mouseBtn, this.mouseStatus, x, y, this);
}

Cell.prototype.mouseMove = function (event) {    
    var x = event.clientX - this.svg.getBoundingClientRect().left;
    var y = event.clientY - this.svg.getBoundingClientRect().top;

    if(this.mouseCurAction.mouseMove(this.mouseBtn, this.mouseStatus, x, y, this.mousePre.x, this.mousePre.y, this)) {
        //reset previous mouse position if move action done
        this.mousePre.x = x;
        this.mousePre.y = y;
    }
}

Cell.prototype.mouseUp = function(event) {
    this.mouseStatus = BTN_UP;
    var x = event.clientX - this.svg.getBoundingClientRect().left;
    var y = event.clientY - this.svg.getBoundingClientRect().top;

    this.mouseCurAction.mouseUp(this.mouseBtn, this.mouseStatus, x, y, this);

    //reset mouse status
    this.mouseBtn = BTN_NONE;
    this.mousePre.x = 0;
    this.mousePre.y = 0;
}

Cell.prototype.activeAction = function(id) {
    if(id == ACTION_ID_MRP_ANNOTATION) {
        this.mouseCurAction = this.mouseActionAnnotation;
    } else {
        this.mouseCurAction = this.mouseActionCommon;
        this.mouseActionCommon.registerOpID(id);
    }
}

Cell.prototype.prepare = function() {
    if(this.svg != null) {
        var mouseDown_ = (function(event) {
            this.mouseDown(event);
        }).bind(this);
        this.svg.addEventListener('mousedown', mouseDown_);

        var mouseMove_ = (function(event) {
            this.mouseMove(event);
        }).bind(this);
        this.svg.addEventListener('mousemove', mouseMove_);

        var mouseUp_ = (function(event) {
            this.mouseUp(event);
        }).bind(this);
        this.svg.addEventListener('mouseup', mouseUp_);

        // add texts at four corners
        var width = this.canvas.width;
        var height = this.canvas.height;
        d3.select(this.svg)
            .selectAll('text')
            .data(['LT', 'LB', 'RT', 'RB'])
            .enter()
            .append("text")
            .attr('font-family', 'monospace')
            .attr('fill', '#dcdcdc')
            .attr('alignment-baseline', function (d, i) {
                return (i % 2 == 0) ? 'hanging' : 'baseline';
            })
            .attr('text-anchor', function (d, i) {
                return (i < 2) ? 'start' : 'end';
            })
            .attr('x', function (d, i) {
                return (i < 2) ? TEXT_MARGIN : width - TEXT_MARGIN;
            })
            .attr('y', function (d, i) {
                return (i % 2 == 0) ? TEXT_MARGIN : height - TEXT_MARGIN;
            })
            .attr('id', function (d) {
                return d;
            })
            .attr('font-size', '15px')
            .attr('class', 'no-select-text');
    }
    this.mouseCurAction = this.mouseActionCommon;
    this.mouseActionCommon.registerOpID(ACTION_ID_NONE);
}