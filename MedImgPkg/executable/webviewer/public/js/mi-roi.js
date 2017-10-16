function ROICircle(key, svg, cx, cy, r){
    this.svg = svg;
    this.cx = cx;
    this.cy = cy;
    this.r = r;

    this.roiMain = null;
    this.roiCtrlLT = null;
    this.roiCtrlLB = null;
    this.roiCtrlRT = null;
    this.roiCtrlRB = null;
    this.roiCtrlMove = null;
    this.dragingCallback = null;
    this.dragEndCallback = null;

    this.key = key;
    this.keyMain = key+'-main';
    this.keyCtrlLT = key+'-lt';
    this.keyCtrlLB = key+'-lb';
    this.keyCtrlRT = key+'-rt';
    this.keyCtrlRB = key+'-rb';
    this.keyCtrlMove = key+'-move';
    this.hovering = false;
    this.ctrlSize = 4;

    //adjust ctrl size
    var width = parseFloat(d3.select(svg).attr('width'));
    var height = parseFloat(d3.select(svg).attr('height'));
    this.resize(width, height);

    var hoveringFunc = (function () {
        if (this.hovering == false) {
            //each of the five circle is visible/draggble
            this.roiCtrlLT.style('cursor', 'move');
            this.roiCtrlLB.style('cursor', 'move');
            this.roiCtrlRT.style('cursor', 'move');
            this.roiCtrlRB.style('cursor', 'move');
            this.roiCtrlMove.style('cursor', 'move');

            this.roiCtrlLT.transition().duration(ROICircle.ctrlAppearInverval).attr('r', this.ctrlSize);
            this.roiCtrlLB.transition().duration(ROICircle.ctrlAppearInverval).attr('r', this.ctrlSize);
            this.roiCtrlRT.transition().duration(ROICircle.ctrlAppearInverval).attr('r', this.ctrlSize);
            this.roiCtrlRB.transition().duration(ROICircle.ctrlAppearInverval).attr('r', this.ctrlSize);
            this.roiCtrlMove.transition().duration(ROICircle.ctrlAppearInverval).attr('r', this.ctrlSize);

            //this.roiMain.style('cursor', 'default');
            this.hovering = true;
        }
    }).bind(this);

    var hoveringDoneFunc = (function () {
        //each of the five circle is invisible none-draggble
        this.roiCtrlLT.style('cursor', 'default');
        this.roiCtrlLB.style('cursor', 'default');
        this.roiCtrlRT.style('cursor', 'default');
        this.roiCtrlRB.style('cursor', 'default');
        this.roiCtrlMove.style('cursor', 'default');

        this.roiCtrlLT.transition().duration(ROICircle.ctrlFadeInverval).attr('r', 0.0);
        this.roiCtrlLB.transition().duration(ROICircle.ctrlFadeInverval).attr('r', 0.0);
        this.roiCtrlRT.transition().duration(ROICircle.ctrlFadeInverval).attr('r', 0.0);
        this.roiCtrlRB.transition().duration(ROICircle.ctrlFadeInverval).attr('r', 0.0);
        this.roiCtrlMove.transition().duration(ROICircle.ctrlFadeInverval).attr('r', 0.0);

        //this.roiMain.style("cursor", "default");
        this.hovering = false;
    }).bind(this);
    
    //SVG circles
    //main contour (selected as hovering area)
    this.roiMain = d3.select(svg).selectAll('circle')
        .data([{
            key: this.keyMain,
            cx: cx,
            cy: cy,
            r: r
        }], function (d) {
            return d.key;
        }).enter().append('circle')
        .attr('cx', function (d) {
            return d.cx;
        })
        .attr('cy', function (d) {
            return d.cy;
        })
        .attr('r', function (d) {
            return d.r;
        })
        // .style('fill', 'none')
        .style('fill-opacity', 0.0)
        .style('stroke', ROICircle.mainColor)
        .style('stroke-opacity', 1.0)
        .style('stroke-width', 2)
        .on("mouseover", hoveringFunc)
        .on("mouseout", hoveringDoneFunc);

    //4 ctrl circle for stretching (selected whole circle)
    this.roiCtrlLT = d3.select(svg).selectAll('circle')
    .data([{key:this.keyCtrlLT, cx:cx, cy:cy, r:r}], function(d) {
        return d.key;
    }).enter().append('circle')
    .attr('cx', function(d) { return Math.floor(d.cx - 0.707*d.r);})
    .attr('cy', function(d) { return Math.floor(d.cy - 0.707*d.r);})
    .attr('r', this.ctrlSize)
    .style('fill', ROICircle.ctrlColor)
    .on("mouseover", hoveringFunc)
    .on("mouseout", hoveringDoneFunc);
    //.style('cursor', 'move');

    this.roiCtrlLB = d3.select(svg).selectAll('circle')
    .data([{key:this.keyCtrlLB, cx:cx, cy:cy, r:r}], function(d) {
        return d.key;
    }).enter().append('circle')
    .attr('cx', function(d) { return Math.floor(d.cx - 0.707*d.r);})
    .attr('cy', function(d) { return Math.floor(d.cy + 0.707*d.r);})
    .attr('r', this.ctrlSize)
    .style('fill', ROICircle.ctrlColor)
    .on("mouseover", hoveringFunc)
    .on("mouseout", hoveringDoneFunc);
    // .style('cursor', 'move');

    this.roiCtrlRT = d3.select(svg).selectAll('circle')
    .data([{key:this.keyCtrlRT, cx:cx, cy:cy, r:r}], function(d) {
        return d.key;
    }).enter().append('circle')
    .attr('cx', function(d) { return Math.floor(d.cx + 0.707*d.r);})
    .attr('cy', function(d) { return Math.floor(d.cy - 0.707*d.r);})
    .attr('r', this.ctrlSize)
    .style('fill', ROICircle.ctrlColor)
    .on("mouseover", hoveringFunc)
    .on("mouseout", hoveringDoneFunc);
    // .style('cursor', 'move');

    this.roiCtrlRB = d3.select(svg).selectAll('circle')
    .data([{key:this.keyCtrlRB, cx:cx, cy:cy, r:r}], function(d) {
        return d.key;
    }).enter().append('circle')
    .attr('cx', function(d) { return Math.floor(d.cx + 0.707*d.r);})
    .attr('cy', function(d) { return Math.floor(d.cy + 0.707*d.r);})
    .attr('r', this.ctrlSize)
    .style('fill', ROICircle.ctrlColor)
    .on("mouseover", hoveringFunc)
    .on("mouseout", hoveringDoneFunc);
    // .style('cursor', 'move');

    //1 ctrl circle for moving (selected loop)
    this.roiCtrlMove = d3.select(svg).selectAll('circle')
    .data([{key:this.keyCtrlMove, cx:cx, cy:cy, r:r}], function(d) {
        return d.key;
    }).enter().append('circle')
    .attr('cx', function(d) { return d.cx;})
    .attr('cy', function(d) { return d.cy;})
    .attr('r', this.ctrlSize)
    .style('fill', ROICircle.ctrlColor)
    .on("mouseover", hoveringFunc)
    .on("mouseout", hoveringDoneFunc);
    // .style('fill-opacity', 0.0)
    // .style('stroke', ROICircle.ctrlColor)
    // .style('stroke-opacity', 1.0)
    // .style('stroke-width', 3);
    // .style('cursor', 'move');

    //dragger
    var dragEndCtrl = (function (d) {
        if (this.dragEndCallback) {
            return this.dragEndCallback(this.cx, this.cy, this.r, this.key);
        }
    }).bind(this);

    this.roiCtrlMove.call(d3.drag().on('drag', (function (d) {
        this.move(d3.event.x, d3.event.y);
        if (this.dragingCallback) {
            this.dragingCallback(this.cx, this.cy, this.r, this.key);
        }
    }).bind(this)).on('end', dragEndCtrl));

    var dragCtrlStretch = (function (d) {
        let cx = this.roiMain.attr('cx');
        let cy = this.roiMain.attr('cy');
        let r = Math.sqrt((d3.event.x - cx) * (d3.event.x - cx) + (d3.event.y - cy) * (d3.event.y - cy));
        this.stretch(Math.floor(r));
        if (this.dragingCallback) {
            return this.dragingCallback(this.cx, this.cy, this.r, this.key);
        }
    }).bind(this);

    //2:register drop operation
    this.roiCtrlLT.call(d3.drag().on('drag', dragCtrlStretch).on('end', dragEndCtrl));
    this.roiCtrlLB.call(d3.drag().on('drag', dragCtrlStretch).on('end', dragEndCtrl));
    this.roiCtrlRT.call(d3.drag().on('drag', dragCtrlStretch).on('end', dragEndCtrl));
    this.roiCtrlRB.call(d3.drag().on('drag', dragCtrlStretch).on('end', dragEndCtrl));
}

ROICircle.mainColor = 'red';
ROICircle.ctrlColor = '#DC143C';
ROICircle.highlightColor = 'yellow';
ROICircle.ctrlAppearInverval = 250;
ROICircle.ctrlFadeInverval = 2000;

ROICircle.prototype.resize = function(w, h) {
    // based on the size of cell-window, tune the ctrl circle radius, but still clamp to [1, 6]
    this.ctrlSize = Math.min(Math.max((w + h) / 500.0, 3.5), 6);
}

ROICircle.prototype.move = function(cx, cy) {
    let r = parseFloat(this.roiMain.attr('r'));
    this.cx = cx;
    this.cy = cy;

    this.roiMain
    .attr('cx', cx)
    .attr('cy', cy);
    this.roiCtrlLT
    .attr('cx', Math.floor(cx - 0.707*r))
    .attr('cy', Math.floor(cy - 0.707*r));
    this.roiCtrlLB
    .attr('cx', Math.floor(cx - 0.707*r))
    .attr('cy', Math.floor(cy + 0.707*r));
    this.roiCtrlRT
    .attr('cx', Math.floor(cx + 0.707*r))
    .attr('cy', Math.floor(cy - 0.707*r));
    this.roiCtrlRB
    .attr('cx', Math.floor(cx + 0.707*r))
    .attr('cy', Math.floor(cy + 0.707*r));
    this.roiCtrlMove
    .attr('cx', cx)
    .attr('cy', cy);
}

ROICircle.prototype.stretch = function(r) {
    let cx = parseFloat(this.roiMain.attr('cx'));
    let cy = parseFloat(this.roiMain.attr('cy'));
    this.r = r;

    this.roiMain
    .attr('r', r);
    this.roiCtrlLT
    .attr('cx', Math.floor(cx - 0.707*r))
    .attr('cy', Math.floor(cy - 0.707*r));
    this.roiCtrlLB
    .attr('cx', Math.floor(cx - 0.707*r))
    .attr('cy', Math.floor(cy + 0.707*r));
    this.roiCtrlRT
    .attr('cx', Math.floor(cx + 0.707*r))
    .attr('cy', Math.floor(cy - 0.707*r));
    this.roiCtrlRB
    .attr('cx', Math.floor(cx + 0.707*r))
    .attr('cy', Math.floor(cy + 0.707*r));
    this.roiCtrlMove
    .attr('cx', cx)
    .attr('cy', cy);
}

ROICircle.prototype.locate = function(cx, cy, r) {
    this.cx = cx;
    this.cy = cy;
    this.r = r;

    this.roiMain
    .attr('cx', cx)
    .attr('cy', cy)
    .attr('r', r);
    this.roiCtrlLT
    .attr('cx', Math.floor(cx - 0.707*r))
    .attr('cy', Math.floor(cy - 0.707*r));
    this.roiCtrlLB
    .attr('cx', Math.floor(cx - 0.707*r))
    .attr('cy', Math.floor(cy + 0.707*r));
    this.roiCtrlRT
    .attr('cx', Math.floor(cx + 0.707*r))
    .attr('cy', Math.floor(cy - 0.707*r));
    this.roiCtrlRB
    .attr('cx', Math.floor(cx + 0.707*r))
    .attr('cy', Math.floor(cy + 0.707*r));
    this.roiCtrlMove
    .attr('cx', cx)
    .attr('cy', cy);
}

ROICircle.prototype.visible = function(flag) {
    if( flag == true) {
        var vis = 'inline';
    } else {
        var vis = 'none';
    }

    this.roiMain.style('display', vis);
    this.roiCtrlLT.style('display', vis);
    this.roiCtrlLB.style('display', vis);
    this.roiCtrlRT.style('display', vis);
    this.roiCtrlRB.style('display', vis);
    this.roiCtrlMove.style('display', vis);
}

ROICircle.prototype.setCtrlRadius = function(radius) {
    this.roiCtrlLT.attr('r', radius);
    this.roiCtrlLB.attr('r', radius);
    this.roiCtrlRT.attr('r', radius);
    this.roiCtrlRB.attr('r', radius);
    this.roiCtrlMove.attr('r', radius);
}

ROICircle.prototype.creating = function(x, y) {
    let cx = parseFloat(this.roiMain.attr('cx'));
    let cy = parseFloat(this.roiMain.attr('cy'));
    let r = Math.sqrt((x - cx)*(x - cx) + (y - cy)*(y - cy));
    this.stretch(Math.floor(r));
}

ROICircle.prototype.release = function() {
    var data = d3.select(this.svg).selectAll('circle').data();
    var newData = [];
    for (var i = 0; i < data.length; ++i) {
        if (data[i].key != this.keyMain &&
            data[i].key != this.keyCtrlLB &&
            data[i].key != this.keyCtrlLT &&
            data[i].key != this.keyCtrlRB &&
            data[i].key != this.keyCtrlRT &&
            data[i].key != this.keyCtrlMove) {
            newData.push(data[i]);
        }
    }

    d3.select(this.svg).selectAll('circle')
    .data(newData, function(d) {
        return d.key;
    }).exit().remove();
}
