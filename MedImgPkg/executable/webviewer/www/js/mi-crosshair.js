

function sendCrosshairMSG(cellID, cx, cy, line0Para, line1Para, socketClient) {
    let buffer = Protobuf.encode(socketClient, 'MsgCrosshair', {
        cx: cx,
        cy: cy,
        l0a: line0Para.a,
        l0b: line0Para.b,
        l1a: line1Para.a,
        l1b: line1Para.b,
    });
    if (buffer) {
        socketClient.sendData(COMMAND_ID_BE_FE_OPERATION, OPERATION_ID_BE_FE_LOCATE, cellID, buffer.byteLength, buffer);
    }
}

function Crosshair(svg, cellID, cx, cy, line0Para, line1Para, socketClient, style) {
    this.svg = svg;
    this.cellID = cellID;
    this.cx = cx;
    this.cy = cy;
    this.line0Para = line0Para;//{a: b: c:} ax + by = c
    this.line1Para = line1Para;//{a: b: c:}
    this.socketClient = socketClient;

    this.style = style; // 0 for MPR 1 for VR
    this.crossContinuous = false;

    //main
    this.line00;
    this.line01;
    this.line10;
    this.line11;
    this.crossUp;
    this.crossDown;
    this.crossLeft;
    this.crossRight;

    //crosshair
    this.crossCtrlOverlay;
    this.crossCtrl;

    //move
    this.line00CtrlMove;
    this.line01CtrlMove;
    this.line10CtrlMove;
    this.line11CtrlMove;

    //style
    this.lineWidth = 1.5;
    this.crossWidth = 2;
    this.lineCtrlWidth = 5;
    this.line0Color ='red';
    this.line1Color = 'yellow';
    this.crossColor = 'blue';
    this.crossSize = 20;
    this.crossCtrlSize = 7;

    //clock
    this.mouseClock = new Date().getTime();

    if (0 == style) {
        this.initMPRStyle();
    } else if (1 == style){
        this.initVRStyle();
    }
}

Crosshair.prototype.initVRStyle = function() {
    this.crossUp = d3.select(this.svg).selectAll('line')
    .data([{key: 'crosshair-cross-up'}], function(d) {return d.key;})
    .enter()
    .append('line')
    .style('stroke-width', this.crossWidth)
    .style('stroke', this.crossColor)
    .style('stroke-opacity', 1.0);

    this.crossDown = d3.select(this.svg).selectAll('line')
    .data([{key: 'crosshair-cross-down'}], function(d) {return d.key;})
    .enter()
    .append('line')
    .style('stroke-width', this.crossWidth)
    .style('stroke', this.crossColor)
    .style('stroke-opacity', 1.0);

    this.crossLeft = d3.select(this.svg).selectAll('line')
    .data([{key: 'crosshair-cross-left'}], function(d) {return d.key;})
    .enter()
    .append('line')
    .style('stroke-width', this.crossWidth)
    .style('stroke', this.crossColor)
    .style('stroke-opacity', 1.0);

    this.crossRight = d3.select(this.svg).selectAll('line')
    .data([{key: 'crosshair-cross-right'}], function(d) {return d.key;})
    .enter()
    .append('line')
    .style('stroke-width', this.crossWidth)
    .style('stroke', this.crossColor)
    .style('stroke-opacity', 1.0);

    this.crossCtrlOverlay = d3.select(this.svg).selectAll('rect')
    .data([{key: 'crosshair-cross-ctrl-overlay'}], function(d) {return d.key;})
    .enter()
    .append('rect')
    .style('fill', 'white')
    .style('opacity', 0.0);

    this.crossCtrl = d3.select(this.svg).selectAll('rect')
    .data([{key: 'crosshair-cross-ctrl'}], function(d) {return d.key;})
    .enter()
    .append('rect')
    .style('fill', 'white')
    .style('opacity', 0.0)
    .style('cursor', 'move');

    this.setCross(this.cx, this.cy, this.line0Para, this.line1Para);

    //drag crosshair
    this.crossCtrl.call(d3.drag().
    on('drag', (function (d) {
        let x = d3.event.x;
        let y = d3.event.y;
        //calculate new line parameter
        this.moveCrosshair(x,y);

        let curClock = new Date().getTime();
        if (Math.abs(this.mouseClock - curClock) < MOUSE_MSG_INTERVAL) {
            return;
        }
        //update mouse clock
        this.mouseClock = curClock;
        
        //send message
        if (this.crossContinuous) {
            this.sendMSG();    
        }
    }).bind(this))
    .on('end', (function (d) {
        let x = d3.event.x;
        let y = d3.event.y;
        //calculate new line parameter
        this.moveCrosshair(x,y);
        //send message
        this.sendMSG();
    }).bind(this)));
}

Crosshair.prototype.initMPRStyle = function() {
    this.line00 = d3.select(this.svg).selectAll('line')
    .data([{key: 'crosshair-line00'}], function(d) {return d.key;})
    .enter()
    .append('line')
    .style('stroke-width', this.lineWidth)
    .style('stroke', this.line0Color)
    .style('stroke-opacity', 0.0);

    this.line01 = d3.select(this.svg).selectAll('line')
    .data([{key: 'crosshair-line01'}], function(d) {return d.key;})
    .enter()
    .append('line')
    .style('stroke-width', this.lineWidth)
    .style('stroke', this.line0Color)
    .style('stroke-opacity', 0.0);

    this.line00CtrlMove = d3.select(this.svg).selectAll('line')
    .data([{key: 'crosshair-line00-ctrl-move'}], function(d) {return d.key;})
    .enter()
    .append('line')
    .style('stroke-width', this.lineCtrlWidth)
    .style('stroke', this.line0Color)
    .style('stroke-opacity', 0.0)
    .style('cursor', 'move');

    this.line01CtrlMove = d3.select(this.svg).selectAll('line')
    .data([{key: 'crosshair-line01-ctrl-move'}], function(d) {return d.key;})
    .enter()
    .append('line')
    .style('stroke-width', this.lineCtrlWidth)
    .style('stroke', this.line0Color)
    .style('stroke-opacity', 0.0)
    .style('cursor', 'move');

    this.line10 = d3.select(this.svg).selectAll('line')
    .data([{key: 'crosshair-line10'}], function(d) {return d.key;})
    .enter()
    .append('line')
    .style('stroke-width', this.lineWidth)
    .style('stroke', this.line1Color)
    .style('stroke-opacity', 0.0);

    this.line11 = d3.select(this.svg).selectAll('line')
    .data([{key: 'crosshair-line11'}], function(d) {return d.key;})
    .enter()
    .append('line')
    .style('stroke-width', this.lineWidth)
    .style('stroke', this.line1Color)
    .style('stroke-opacity', 0.0);

    this.line10CtrlMove = d3.select(this.svg).selectAll('line')
    .data([{key: 'crosshair-line10-ctrl-move'}], function(d) {return d.key;})
    .enter()
    .append('line')
    .style('stroke-width', this.lineCtrlWidth)
    .style('stroke', this.line1Color)
    .style('stroke-opacity', 0.0)
    .style('cursor', 'move');

    this.line11CtrlMove = d3.select(this.svg).selectAll('line')
    .data([{key: 'crosshair-line11-ctrl-move'}], function(d) {return d.key;})
    .enter()
    .append('line')
    .style('stroke-width', this.lineCtrlWidth)
    .style('stroke', this.line1Color)
    .style('stroke-opacity', 0.0)
    .style('cursor', 'move');

    this.crossUp = d3.select(this.svg).selectAll('line')
    .data([{key: 'crosshair-cross-up'}], function(d) {return d.key;})
    .enter()
    .append('line')
    .style('stroke-width', this.crossWidth)
    .style('stroke', this.crossColor)
    .style('stroke-opacity', 0.0);

    this.crossDown = d3.select(this.svg).selectAll('line')
    .data([{key: 'crosshair-cross-down'}], function(d) {return d.key;})
    .enter()
    .append('line')
    .style('stroke-width', this.crossWidth)
    .style('stroke', this.crossColor)
    .style('stroke-opacity', 0.0);

    this.crossLeft = d3.select(this.svg).selectAll('line')
    .data([{key: 'crosshair-cross-left'}], function(d) {return d.key;})
    .enter()
    .append('line')
    .style('stroke-width', this.crossWidth)
    .style('stroke', this.crossColor)
    .style('stroke-opacity', 0.0);

    this.crossRight = d3.select(this.svg).selectAll('line')
    .data([{key: 'crosshair-cross-right'}], function(d) {return d.key;})
    .enter()
    .append('line')
    .style('stroke-width', this.crossWidth)
    .style('stroke', this.crossColor)
    .style('stroke-opacity', 0.0);

    this.crossCtrlOverlay = d3.select(this.svg).selectAll('rect')
    .data([{key: 'crosshair-cross-ctrl-overlay'}], function(d) {return d.key;})
    .enter()
    .append('rect')
    .style('fill', 'white')
    .style('opacity', 0);

    this.crossCtrl = d3.select(this.svg).selectAll('rect')
    .data([{key: 'crosshair-cross-ctrl'}], function(d) {return d.key;})
    .enter()
    .append('rect')
    .style('fill', 'white')
    .style('opacity', 0)
    .style('cursor', 'move');

    this.setLine(this.cx, this.cy, this.line0Para, this.line1Para);

    //drag crosshair
    this.crossCtrl.call(d3.drag().
    on('drag', (function (d) {
        let x = d3.event.x;
        let y = d3.event.y;
        //calculate new line parameter
        this.moveCrosshair(x,y);

        let curClock = new Date().getTime();
        if (Math.abs(this.mouseClock - curClock) < MOUSE_MSG_INTERVAL) {
            return;
        }
        //update mouse clock
        this.mouseClock = curClock;

        //send message
        if (this.crossContinuous) {
            this.sendMSG();
        }
    }).bind(this))
    .on('end', (function (d) {
        let x = d3.event.x;
        let y = d3.event.y;
        //calculate new line parameter
        this.moveCrosshair(x,y);
        //send message
        this.sendMSG();
    }).bind(this)));

    this.line00CtrlMove.call(d3.drag()
    .on('drag', (function(d) {
        let x = d3.event.x;
        let y = d3.event.y;
        //calculate new line parameter
        this.moveLine0(x,y);

        let curClock = new Date().getTime();
        if (Math.abs(this.mouseClock - curClock) < MOUSE_MSG_INTERVAL) {
            return;
        }
        //update mouse clock
        this.mouseClock = curClock;
        //send message
        this.sendMSG();

    }).bind(this))
    .on('end', (function (d) {
        let x = d3.event.x;
        let y = d3.event.y;
        //calculate new line parameter
        this.moveLine0(x,y);
        //send message
        this.sendMSG();
    }).bind(this)));

    this.line01CtrlMove.call(d3.drag()
    .on('drag', (function(d) {
        let x = d3.event.x;
        let y = d3.event.y;
        //calculate new line parameter
        this.moveLine0(x,y);

        let curClock = new Date().getTime();
        if (Math.abs(this.mouseClock - curClock) < MOUSE_MSG_INTERVAL) {
            return;
        }
        //update mouse clock
        this.mouseClock = curClock;
        //send message
        this.sendMSG();

    }).bind(this))
    .on('end', (function (d) {
        let x = d3.event.x;
        let y = d3.event.y;
        //calculate new line parameter
        this.moveLine0(x,y);
        //send message
        this.sendMSG();
    }).bind(this)));


    this.line10CtrlMove.call(d3.drag()
    .on('drag', (function(d) {
        let x = d3.event.x;
        let y = d3.event.y;
        //calculate new line parameter
        this.moveLine1(x,y);

        let curClock = new Date().getTime();
        if (Math.abs(this.mouseClock - curClock) < MOUSE_MSG_INTERVAL) {
            return;
        }
        //update mouse clock
        this.mouseClock = curClock;
        //send message
        this.sendMSG();

    }).bind(this))
    .on('end', (function (d) {
        let x = d3.event.x;
        let y = d3.event.y;
        //calculate new line parameter
        this.moveLine1(x,y);
        //send message
        this.sendMSG();
    }).bind(this)));

    this.line11CtrlMove.call(d3.drag()
    .on('drag', (function(d) {
        let x = d3.event.x;
        let y = d3.event.y;
        //calculate new line parameter
        this.moveLine1(x,y);

        let curClock = new Date().getTime();
        if (Math.abs(this.mouseClock - curClock) < MOUSE_MSG_INTERVAL) {
            return;
        }
        //update mouse clock
        this.mouseClock = curClock;
        //send message
        this.sendMSG();

    }).bind(this))
    .on('end', (function (d) {
        let x = d3.event.x;
        let y = d3.event.y;
        //calculate new line parameter
        this.moveLine1(x,y);
        //send message
        this.sendMSG();
    }).bind(this)));
}


function distance(p0, p1) {
    return Math.sqrt((p0.x - p1.x)*(p0.x - p1.x) + (p0.y - p1.y)*(p0.y - p1.y));
}

function dot(a, b) {
    return a.x*b.x + a.y*b.y;
}

Crosshair.prototype.moveCrosshair = function(x, y) {
    if (this.style == 0) {
        let para = this.line0Para;
        let ab = Math.sqrt(para.a*para.a + para.b*para.b);
        let normx = para.a/ab;
        let normy = para.b/ab;
        let dis = dot({x:x-this.cx, y:y-this.cy}, {x:normx, y:normy});
        let newc = (para.c/ab + dis)*ab;
        let newpara0 = {a:para.a, b:para.b, c:newc };  
    
        para = this.line1Para;
        ab = Math.sqrt(para.a*para.a + para.b*para.b);
        normx = para.a/ab;
        normy = para.b/ab;
        dis = dot({x:x-this.cx, y:y-this.cy}, {x:normx, y:normy});
        newc = (para.c/ab + dis)*ab;
        let newpara1 = {a:para.a, b:para.b, c:newc };  
    
        this.setLine(x, y, newpara0, newpara1);
    } else if (this.style == 1) {
        this.setCross(x, y);
    }
}

Crosshair.prototype.moveLine0 = function(x, y) {
    if (this.style != 0) {
        return;
    }
    let para = this.line0Para;
    let ab = Math.sqrt(para.a*para.a + para.b*para.b);
    let normx = para.a/ab;
    let normy = para.b/ab;
    let dis = dot({x:x-this.cx, y:y-this.cy}, {x:normx, y:normy});
    let newcx = this.cx + dis*normx;
    let newcy = this.cy + dis*normy;
    let newc = (para.c/ab + dis)*ab;
    let newpara = {a:para.a, b:para.b, c:newc};  

    this.setLine(newcx, newcy, newpara, this.line1Para);
}

Crosshair.prototype.moveLine1 = function(x, y) {
    if (this.style != 0) {
        return;
    }
    let para = this.line1Para;
    let ab = Math.sqrt(para.a*para.a + para.b*para.b);
    let normx = para.a/ab;
    let normy = para.b/ab;
    let dis = dot({x:x-this.cx, y:y-this.cy}, {x:normx, y:normy});
    let newcx = this.cx + dis*normx;
    let newcy = this.cy + dis*normy;
    let newc = (para.c/ab + dis)*ab;
    let newpara = {a:para.a, b:para.b, c:newc};  

    this.setLine(newcx, newcy, this.line0Para, newpara);
}

Crosshair.prototype.calLine = function(cx, cy, para) {
    const width = $(this.svg).attr('width');
    const height = $(this.svg).attr('height');

    //cross 4 border
    //TODO a b == 0
    let res = [];
    let x0 = 0;
    let y0 = (para.c - para.a * x0) / para.b;
    if (y0 >= 0 && y0 <= height - 1) { res.push({x:x0, y:y0});}
    let x1 = width - 1;
    let y1 = (para.c - para.a * x1) / para.b;
    if (y1 >= 0 && y1 <= height - 1) { res.push({x:x1, y:y1});}
    let y2 = 0;
    let x2 = (para.c - para.b * y2) / para.a;
    if (x2 >= 0 && x2 <= width - 1) { res.push({x:x2, y:y2});}
    let y3 = height - 1;
    let x3 = (para.c - para.b * y3) / para.a;
    if (x3 >= 0 && x3 <= width - 1) { res.push({x:x3, y:y3});}
    if (res.length != 2)  {
        return [{x:0, y:0}, {x:0,y:0}, {x:0, y:0}, {x:0,y:0}];
    }

    let ab = Math.sqrt(para.a*para.a + para.b*para.b);
    let dx = para.b/ab;
    let dy = para.a/ab;
    let p0 = {x:cx - this.crossSize*dx, y:cy - this.crossSize*dy};
    let p1 = {x:cx + this.crossSize*dx, y:cy + this.crossSize*dy};

    if (distance(res[0], p0) <= distance(res[0], {x:this.cx, y:this.cy})) {
        return [res[0], p0, res[1], p1];
    } else {
        return [res[0], p1, res[1], p0];
    }
}

Crosshair.prototype.setCross = function(cx, cy) {
    this.cx = cx;
    this.cy = cy;

    this.crossLeft
    .attr('x1', this.cx)
    .attr('y1', this.cy)
    .attr('x2', this.cx - this.crossSize/2)
    .attr('y2', this.cy);

    this.crossRight
    .attr('x1', this.cx)
    .attr('y1', this.cy)
    .attr('x2', this.cx + this.crossSize/2)
    .attr('y2', this.cy);

    this.crossUp
    .attr('x1', this.cx)
    .attr('y1', this.cy)
    .attr('x2', this.cx)
    .attr('y2', this.cy - this.crossSize/2);

    this.crossDown
    .attr('x1', this.cx)
    .attr('y1', this.cy)
    .attr('x2', this.cx)
    .attr('y2', this.cy + this.crossSize/2);

    this.crossCtrlOverlay
    .attr('x', cx - this.crossSize/2)
    .attr('y', cy - this.crossSize/2)
    .attr('width', this.crossSize)
    .attr('height', this.crossSize);

    this.crossCtrl
    .attr('x', cx - this.crossCtrlSize/2)
    .attr('y', cy - this.crossCtrlSize/2)
    .attr('width', this.crossCtrlSize)
    .attr('height', this.crossCtrlSize);
}

Crosshair.prototype.setLine = function(cx, cy, para0, para1) {
    this.cx = cx;
    this.cy = cy;
    this.line0Para = para0;//{a: b: c:} ax + by = c
    this.line1Para = para1;

    let l0 = this.calLine(this.cx, this.cy, this.line0Para);
    this.line00
    .attr('x1', l0[0].x)
    .attr('y1', l0[0].y)
    .attr('x2', l0[1].x)
    .attr('y2', l0[1].y);

    this.line01
    .attr('x1', l0[2].x)
    .attr('y1', l0[2].y)
    .attr('x2', l0[3].x)
    .attr('y2', l0[3].y);

    this.line00CtrlMove
    .attr('x1', l0[0].x)
    .attr('y1', l0[0].y)
    .attr('x2', l0[1].x)
    .attr('y2', l0[1].y);

    this.line01CtrlMove
    .attr('x1', l0[2].x)
    .attr('y1', l0[2].y)
    .attr('x2', l0[3].x)
    .attr('y2', l0[3].y);

    let l1 = this.calLine(this.cx, this.cy, this.line1Para);
    this.line10
    .attr('x1', l1[0].x)
    .attr('y1', l1[0].y)
    .attr('x2', l1[1].x)
    .attr('y2', l1[1].y);

    this.line11
    .attr('x1', l1[2].x)
    .attr('y1', l1[2].y)
    .attr('x2', l1[3].x)
    .attr('y2', l1[3].y);

    this.line10CtrlMove
    .attr('x1', l1[0].x)
    .attr('y1', l1[0].y)
    .attr('x2', l1[1].x)
    .attr('y2', l1[1].y);

    this.line11CtrlMove
    .attr('x1', l1[2].x)
    .attr('y1', l1[2].y)
    .attr('x2', l1[3].x)
    .attr('y2', l1[3].y);

    this.crossLeft
    .attr('x1', this.cx)
    .attr('y1', this.cy)
    .attr('x2', this.cx - this.crossSize/2)
    .attr('y2', this.cy);

    this.crossRight
    .attr('x1', this.cx)
    .attr('y1', this.cy)
    .attr('x2', this.cx + this.crossSize/2)
    .attr('y2', this.cy);

    this.crossUp
    .attr('x1', this.cx)
    .attr('y1', this.cy)
    .attr('x2', this.cx)
    .attr('y2', this.cy - this.crossSize/2);

    this.crossDown
    .attr('x1', this.cx)
    .attr('y1', this.cy)
    .attr('x2', this.cx)
    .attr('y2', this.cy + this.crossSize/2);

    this.crossCtrlOverlay
    .attr('x', cx - this.crossSize/2)
    .attr('y', cy - this.crossSize/2)
    .attr('width', this.crossSize)
    .attr('height', this.crossSize);

    this.crossCtrl
    .attr('x', cx - this.crossCtrlSize/2)
    .attr('y', cy - this.crossCtrlSize/2)
    .attr('width', this.crossCtrlSize)
    .attr('height', this.crossCtrlSize);
}

Crosshair.prototype.sendMSG = function() {
    sendCrosshairMSG(this.cellID, this.cx, this.cy, this.line0Para, this.line1Para, this.socketClient);
}

Crosshair.prototype.parseNoneImg = function (msgCrosshair) {
    if (!msgCrosshair) {
        return;
    }

    if (this.style == 0) {
        let cx = msgCrosshair.cx;
        let cy = msgCrosshair.cy;
        let para0 = {a:msgCrosshair.l0A, b:msgCrosshair.l0B, c:msgCrosshair.l0C};
        let para1 = {a:msgCrosshair.l1A, b:msgCrosshair.l1B, c:msgCrosshair.l1C};
        if (msgCrosshair.l0Color) {
            if (this.line00) {
                this.line00.style('stroke', msgCrosshair.l0Color)
                .style('stroke-opacity',1.0);
            }
    
            if (this.line01) {
                this.line01.style('stroke', msgCrosshair.l0Color)
                .style('stroke-opacity',1.0);
            }
    
            //this.crossUp.style('stroke-opacity', 1.0);
            //this.crossDown.style('stroke-opacity', 1.0);
            //this.crossLeft.style('stroke-opacity', 1.0);
            //this.crossRight.style('stroke-opacity', 1.0);
        }
        if (msgCrosshair.l1Color) {
            if (this.line10) {
                this.line10.style('stroke', msgCrosshair.l1Color)
                .style('stroke-opacity',1.0);
            }
            if (this.line11) {
                this.line11.style('stroke', msgCrosshair.l1Color)
                .style('stroke-opacity',1.0);
            }
            //this.crossUp.style('stroke-opacity', 1.0);
            //this.crossDown.style('stroke-opacity', 1.0);
            //this.crossLeft.style('stroke-opacity', 1.0);
            //this.crossRight.style('stroke-opacity', 1.0);
        }
        this.setLine(cx, cy, para0, para1);
    } else if (this.style == 1) {
        let cx = msgCrosshair.cx;
        let cy = msgCrosshair.cy;
        this.setCross(cx, cy);
    }   
}

Crosshair.prototype.visible = function(flag) {
    let vis = 'none';
    if( flag == true) {
        vis = 'inline';
    } else {
        vis = 'none';
    }
    if (this.style == 0) {
        this.line00.style('display', vis);
        this.line01.style('display', vis);
        this.line00CtrlMove.style('display', vis);
        this.line01CtrlMove.style('display', vis);
        this.line10.style('display', vis);
        this.line11.style('display', vis);
        this.line10CtrlMove.style('display', vis);
        this.line11CtrlMove.style('display', vis);
        this.crossUp.style('display', vis);
        this.crossDown.style('display', vis);
        this.crossLeft.style('display', vis);
        this.crossRight.style('display', vis);
        this.crossCtrl.style('display', vis);
        this.crossCtrlOverlay.style('display', vis);
    } else if (this.style == 1) {
        this.crossUp.style('display', vis);
        this.crossDown.style('display', vis);
        this.crossLeft.style('display', vis);
        this.crossRight.style('display', vis);
        this.crossCtrl.style('display', vis);
        this.crossCtrlOverlay.style('display', vis);
    }
}