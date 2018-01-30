//corner info marin
const TEXT_MARGIN = 4;
const DOUBLE_CLICK_INTERVAL = 250;
const HOLD_CLICK_INTERVAL = 100;
const COORDINATE_LABELS = ['L', 'R', 'H', 'F', 'P', 'A'];

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
    
    this.jpegImg.onload = (function(){
        refreshCanvas(this.canvas, this.jpegImg);
    }).bind(this);

    //mouse event
    this.mouseBtn = BTN_NONE;
    this.mouseStatus = BTN_UP;
    this.mouseCur = {x:0,y:0};//for mouse holder
    this.mousePre = {x:0,y:0};
    this.mouseClickTick = 0;
    this.mouseHoldTick = BTN_NONE;

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
    this.borderColor = '#333333';

    this.ruler = null;

    //mouse out event
    $(this.svg).mouseleave((function() {
        if(this.mouseStatus == BTN_DOWN) {
            //do mouse up logic
            this.mouseStatus = BTN_UP;
            this.mouseCurAction.mouseUp(this.mouseBtn, this.mouseStatus, this.mousePre.x, this.mousePre.y, this);
            //reset mouse
            this.mouseBtn = BTN_NONE;
            this.mousePre.x = 0;
            this.mousePre.y = 0;
        }
    }).bind(this));
}

Cell.prototype.release = function() {
    $(this.svg).empty();
}

function refreshCanvas(canvas, img) {
    canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
}

Cell.prototype.handleJpegBuffer = function(buffer, bufferOffset, dataLen, restDataLen, withHeader) {
    if (withHeader) {
        this.jpegStr = '';
    } 
    let imgBuffer = new Uint8Array(buffer, bufferOffset, dataLen);
    this.jpegStr += String.fromCharCode.apply(null, imgBuffer);
    if(restDataLen <= 0) {
        this.jpegImg.src =  'data:image/jpg;base64,' + btoa(this.jpegStr);
    }
}

Cell.prototype.handleNongImgBuffer = function (arrayBuffer) {
    let receivedMsg = Protobuf.decode(this.socketClient, 'MsgNoneImgCollection', arrayBuffer);
    if (!receivedMsg) {
        return;
    }

    if (receivedMsg.hasOwnProperty('cornerInfos')) {
        let txt = receivedMsg.cornerInfos.infos;//MSG Format "LT|1:patientName|2:patientID\nLB....\nRT|....\nRB|....\n"
        let corners = txt.split('\n');

        // for each corner
        // 'length-1': avoid the last empty string "" after '\n'
        for (let i = 0; i < (corners.length - 1); ++i) {
            let oneCornerInfo = corners[i].split('|');
            let cornerTag = oneCornerInfo[0];
            let infoContext = [];
            for (let j = 1; j < oneCornerInfo.length; ++j) {
                let info = oneCornerInfo[j].split(',');
                infoContext.push({pos:info[0], context:info[1]});
            }
            
            let svgText = d3.select(this.svg).select('#' + cornerTag);
            let xpos = null;
            let txtspacing = null;
            let texorigin = null;
            switch(cornerTag) {
                case 'LT':
                    xpos = TEXT_MARGIN;
                    txtspacing = parseFloat(svgText.attr('font-size'));
                    texorigin = 1.2*txtspacing;
                    break;
                case 'RT':
                    xpos = this.canvas.width - TEXT_MARGIN;
                    txtspacing = parseFloat(svgText.attr('font-size'));
                    texorigin = 1.2*txtspacing;
                    break;
                
                case 'LB':
                    xpos = TEXT_MARGIN;
                    txtspacing = -parseFloat(svgText.attr('font-size'));
                    texorigin = this.canvas.height + 0.2*txtspacing;
                    break;
                case 'RB':
                    xpos = this.canvas.width - TEXT_MARGIN;
                    txtspacing = -parseFloat(svgText.attr('font-size'));
                    texorigin = this.canvas.height + 0.2*txtspacing;
                    break;
            }                

            let tspans = svgText.selectAll('tspan');
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
    
    if (receivedMsg.hasOwnProperty('annotations')) {
        let annotations = receivedMsg.annotations.annotation;
        if (annotations) {
            for (let i = 0; i < annotations.length; ++i) {
                let annoUnit = annotations[i];
                let id = annoUnit.id;
                let status = annoUnit.status;
                let vis = annoUnit.visibility;
                let cx = annoUnit.para0;
                let cy = annoUnit.para1;
                let r = annoUnit.para2;
                let content = annoUnit.info;
                let probability = annoUnit.probability;
                switch (annoUnit.status) {
                    case 0: //add
                        this.rois.push(this.mouseActionAnnotation.createROICircle(id, this.svg, cx, cy, r, (vis!=0), probability, content));
                        break;
                    case 1: //delete
                        for (let j = 0; j < this.rois.length; ++j) {
                            if (this.rois[j].key == id) {
                                this.rois[j].release();
                                this.rois.splice(j, 1);
                                break;
                            }
                        }
                        break;
                    case 2: //modifying
                        for (let j = 0; j < this.rois.length; ++j) {
                            if (this.rois[j].key == id) {
                                this.rois[j].locate(cx, cy, r);
                                this.rois[j].visible(vis);
                                this.rois[j].updateContent(content);
                                break;
                            }
                        }
                    break;
                }                    
            }
        }
    }

    //TODO crosshair message
    if (receivedMsg.hasOwnProperty('crosshair')) {
        if (this.crosshair) {
            this.crosshair.parseNoneImg(receivedMsg.crosshair);
            if (receivedMsg.crosshair.borderColor) {
                this.borderColor = receivedMsg.crosshair.borderColor;
                this.canvas.style.border = '3px solid ' + this.borderColor;
            }
        }
    }

    if (receivedMsg.hasOwnProperty('direction')) {
        //console.log('cellID:' + this.cellID + '  ' + receivedMsg.direction.info);
        let txt = receivedMsg.direction.info.split('|'); // format "A|P|H|F"
        let keyTxtArray = [];
        let directionKey = this.key + '-direction';
        for(let i=0; i<txt.length; ++i)
        {
            keyTxtArray.push({key: directionKey, content: txt[i]});
        }
        let directionInfoTxt =
            d3.select(this.svg).selectAll('text').filter(function (d) {
                return (d && d.key) ? d.key == directionKey : false;
            });
        let width = parseFloat(this.svg.getAttribute('width'));
        let height = parseFloat(this.svg.getAttribute('height'));
        if (directionInfoTxt.empty()) {
            directionInfoTxt.data(keyTxtArray)
                .enter()
                .append('text')
                .attr('font-family', 'monospace')
                .attr('font-size', '15px')
                .attr('fill', '#dcdcdc')
                .attr('class', 'no-select-text')
                .attr('alignment-baseline', function (d, i) {
                    switch (i) {
                        case 0:
                            return 'central';
                        case 1:
                            return 'central';
                        case 2:
                            return 'hanging';
                        case 3:
                            return 'baseline';
                    };
                })
                .attr('text-anchor', function (d, i) {
                    switch (i) {
                        case 0:
                            return 'start';
                        case 1:
                            return 'end';
                        case 2:
                            return 'middle';
                        case 3:
                            return 'middle';
                    };
                })
                .attr('x', function (d, i) {
                    switch (i) {
                        case 0:
                            return 0;
                        case 1:
                            return width;
                        case 2:
                            return width*0.5;
                        case 3:
                            return width*0.5;
                    };
                })
                .attr('y', function (d, i) {
                    switch (i) {
                        case 0:
                            return height*0.5;
                        case 1:
                            return height*0.5;
                        case 2:
                            return 0;
                        case 3:
                            return height;
                    };
                })
                .text(function (d) {
                    return d.content;
                });
        } else {
            directionInfoTxt.data(keyTxtArray)
                .attr('x', function (d, i) {
                    switch (i) {
                        case 0:
                            return 0;
                        case 1:
                            return width;
                        case 2:
                            return width * 0.5;
                        case 3:
                            return width * 0.5;
                    };
                })
                .attr('y', function (d, i) {
                    switch (i) {
                        case 0:
                            return height * 0.5;
                        case 1:
                            return height * 0.5;
                        case 2:
                            return 0;
                        case 3:
                            return height;
                    };
                }).text(function (d) {
                    return d.content;
                });
        }
    }

    // frustum information to adapt vertical ruler
    if (receivedMsg.hasOwnProperty('frustum')) {
        if (this.ruler) {
            this.ruler.updateRuler(receivedMsg.frustum.height);
        }
    }
}

Cell.prototype.resize = function (width, height) {
    //canvas&svg resize
    //send msg to notigy BE resize will be call in main
    this.canvas.width = Math.floor(width);
    this.canvas.height = Math.floor(height);
    let top = this.canvas.offsetTop;
    let left = this.canvas.offsetLeft;
    let viewBox = left.toString() + ' ' + top + ' ' + Math.floor(width) + ' ' + Math.floor(height);
    this.svg.setAttribute('viewBox', viewBox);
    this.svg.setAttribute('width', Math.floor(width));
    this.svg.setAttribute('height', Math.floor(height));
    this.svg.setAttribute('x', left);
    this.svg.setAttribute('y', top);


    // resize the txts located at 4 corners
    d3.select(this.svg)
        .selectAll('text').filter(function(d){
            return (d == 'LT' || d == 'LB' || d == 'RT' || d=='RB');
         })
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
                    let txtspacing = parseFloat(d3.select('text').attr('font-size'));
                    if (i % 2 == 0) {
                        let texorigin = 1.2 * txtspacing;
                        return (texorigin + datum.pos * txtspacing);
                    } else {
                        let texorigin = height + 0.2 * (-txtspacing);
                        return (texorigin + datum.pos * (-txtspacing));
                    }
                })
        });
    
    for (let i = 0; i < this.rois.length; ++i) {
        this.rois[i].adjustCircleRadius(width, height);
    }

    // resize the direciton information
    let directionKey = this.key + '-direction';
    let directionInfoTxt =
        d3.select(this.svg).selectAll('text').filter(function (d) {
            return (d && d.key) ? d.key == directionKey : false;
        }).attr('x', function (d, i) {
            switch (i) {
                case 0:
                    return 0;
                case 1:
                    return width;
                case 2:
                    return width * 0.5;
                case 3:
                    return width * 0.5;
            };
        })
        .attr('y', function (d, i) {
            switch (i) {
                case 0:
                    return height * 0.5;
                case 1:
                    return height * 0.5;
                case 2:
                    return 0;
                case 3:
                    return height;
            };
        });
}

Cell.prototype.mouseClickTicker = function() {
    let clickTick = this.mouseClickTick;
    this.mouseClickTick = 0;//reset tick
    if (clickTick >= 2) {
        //double click event
        if (this.mouseDoubleClickEvent) {
            this.mouseDoubleClickEvent(this.cellID);
        }
    }
}

Cell.prototype.touchDown = function(event) {
    event = event || window.event;
    event.preventDefault();

    this.mouseStatus = BTN_DOWN;
    this.mouseBtn = BTN_LEFT;
    this.mouseClickTick += 1;//mouse click tick for check double click
    if (!this.mouseFocus && this.mouseFocusEvent) {
        this.mouseFocusEvent(this.cellID);
    }

    let x = event.touches[0].clientX - this.svg.getBoundingClientRect().left;
    let y = event.touches[0].clientY - this.svg.getBoundingClientRect().top;
    this.mousePre.x = x;
    this.mousePre.y = y;

    setTimeout((function(event) {
        this.mouseClickTicker();
    }).bind(this), DOUBLE_CLICK_INTERVAL);

    this.mouseCurAction.mouseDown(this.mouseBtn, this.mouseStatus, x, y, this);
}

Cell.prototype.touchMove = function (event) { 
    event = event || window.event;
    event.preventDefault();

    let x = event.touches[0].clientX - this.svg.getBoundingClientRect().left;
    let y = event.touches[0].clientY - this.svg.getBoundingClientRect().top;

    if(this.mouseCurAction.mouseMove(this.mouseBtn, this.mouseStatus, x, y, this.mousePre.x, this.mousePre.y, this)) {
        //reset previous mouse position if move action done
        this.mousePre.x = x;
        this.mousePre.y = y;
    }
}

Cell.prototype.touchUp = function(event) {
    event = event || window.event;
    event.preventDefault();

    this.mouseStatus = BTN_UP;
    let x = event.changedTouches[0].clientX - this.svg.getBoundingClientRect().left;
    let y = event.changedTouches[0].clientY - this.svg.getBoundingClientRect().top;

    this.mouseCurAction.mouseUp(this.mouseBtn, this.mouseStatus, x, y, this);

    //reset mouse status
    this.mouseBtn = BTN_NONE;
    this.mousePre.x = 0;
    this.mousePre.y = 0;
}

Cell.prototype.mouseHoldTicker = function() {
    if (this.mouseHoldTick == BTN_LEFT) {
        this.mouseCurAction.mouseDown(this.mouseBtn, this.mouseStatus, this.mouseCur.x, this.mouseCur.y, this);
    } else {
        this.mouseActionCommon.mouseDown(this.mouseBtn, this.mouseStatus, this.mouseCur.x, this.mouseCur.y, this);
    }
    this.mouseBtn = this.mouseHoldTick;
}

function Pow(base, power) {
    let number = base;
    if(power == 1) return number;
    if(power == 0) return 1;
    for(let i = 2; i <= power; i++){
      number = number * base;
    }
    return number;
}

Cell.prototype.mouseDown = function(event) {
    this.mouseStatus = BTN_DOWN;
    //this.mouseBtn = event.button;
    let btn = Pow(2,event.button);
    if (btn == BTN_LEFT) {
        this.mouseClickTick += 1;//left mouse click tick for check double click
    }
    this.mouseHoldTick |= btn;

    if (!this.mouseFocus && this.mouseFocusEvent) {
        this.mouseFocusEvent(this.cellID);
    }

    let x = event.clientX - this.svg.getBoundingClientRect().left;
    let y = event.clientY - this.svg.getBoundingClientRect().top;
    this.mousePre.x = x;
    this.mousePre.y = y;
    this.mouseCur.x = x;
    this.mouseCur.y = y;

    setTimeout((function(event) {
        this.mouseClickTicker();
    }).bind(this), DOUBLE_CLICK_INTERVAL);

    setTimeout((function(event) {
        this.mouseHoldTicker();
    }).bind(this), HOLD_CLICK_INTERVAL);
}

Cell.prototype.mouseMove = function (event) {    
    let x = event.clientX - this.svg.getBoundingClientRect().left;
    let y = event.clientY - this.svg.getBoundingClientRect().top;

    if (this.mouseBtn == BTN_LEFT) {
        if(this.mouseCurAction.mouseMove(this.mouseBtn, this.mouseStatus, x, y, this.mousePre.x, this.mousePre.y, this)) {
            //reset previous mouse position if move action done
            this.mousePre.x = x;
            this.mousePre.y = y;
        }
    } else {
        if(this.mouseActionCommon.mouseMove(this.mouseBtn, this.mouseStatus, x, y, this.mousePre.x, this.mousePre.y, this)) {
            //reset previous mouse position if move action done
            this.mousePre.x = x;
            this.mousePre.y = y;
        }
    }    
}

Cell.prototype.mouseUp = function(event) {
    this.mouseStatus = BTN_UP;
    let x = event.clientX - this.svg.getBoundingClientRect().left;
    let y = event.clientY - this.svg.getBoundingClientRect().top;

    if (this.mouseBtn == BTN_LEFT) {
        this.mouseCurAction.mouseUp(this.mouseBtn, this.mouseStatus, x, y, this);
    } else {
        this.mouseActionCommon.mouseUp(this.mouseBtn, this.mouseStatus, x, y, this);
    }

    //reset mouse status
    this.mouseBtn = BTN_NONE;
    this.mouseHoldTick = BTN_NONE;
    this.mousePre.x = 0;
    this.mousePre.y = 0;
}

Cell.prototype.activeAction = function(left, right, mid, hold) {
    if(left == ACTION_ID_MRP_ANNOTATION) {
        this.mouseCurAction = this.mouseActionAnnotation;
    } else {
        this.mouseCurAction = this.mouseActionCommon;
        this.mouseActionCommon.registerOpID(left, right, mid, hold);
    }
}

Cell.prototype.prepare = function() {
    if(this.svg != null) {
        this.svg.onmousedown = (function(event) {
            this.mouseDown(event);
        }).bind(this);
        this.svg.ontouchstart = (function(event) {
            this.touchDown(event);
        }).bind(this);

        this.svg.onmousemove = (function(event) {
            this.mouseMove(event);
        }).bind(this);
        this.svg.ontouchmove = (function(event) {
            this.touchMove(event);
        }).bind(this)

        this.svg.onmouseup = (function(event) {
            this.mouseUp(event);
        }).bind(this);
        this.svg.ontouchend = (function(event) {
            this.touchUp(event);
        }).bind(this);

        // add texts at four corners
        let width = this.canvas.width;
        let height = this.canvas.height;
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