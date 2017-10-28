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
    this.roiLabel = null;
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
    this.winSize = {width: 0, height: 0};

    //adjust ctrl size
    const width = parseFloat(d3.select(svg).attr('width'));
    const height = parseFloat(d3.select(svg).attr('height'));
    this.adjustCircleRadius(width, height);

    let hoveringFunc = (function () {
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

    let hoveringDoneFunc = (function () {
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
    let dragEndCtrl = (function (d) {
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

    let dragCtrlStretch = (function (d) {
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

ROICircle.prototype.adjustCircleRadius = function(w, h) {
    // based on the size of cell-window, tune the ctrl circle radius, but still clamp to [3.5, 6]
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

    if (this.roiLabel != null)
    {
        this.roiLabel.updateLayout();
    }
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

    if (this.roiLabel != null)
    {
        this.roiLabel.updateLayout();
    }
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

    if (this.roiLabel != null) {
        const width = this.svg.getAttribute("width");
        const height = this.svg.getAttribute("height");

        if(this.winSize.width != width || this.winSize.height != height) {
            if(this.roiLabel.resetLayout()) {
                this.winSize.width = width;
                this.winSize.height = height;
            }
        } else {
            this.roiLabel.updateLayout();
        }
    }
}

ROICircle.prototype.visible = function(flag) {
    let vis = 'inline';
    if( flag == true) {
        vis = 'inline';
    } else {
        vis = 'none';
    }

    this.roiMain.style('display', vis);
    this.roiCtrlLT.style('display', vis);
    this.roiCtrlLB.style('display', vis);
    this.roiCtrlRT.style('display', vis);
    this.roiCtrlRB.style('display', vis);
    this.roiCtrlMove.style('display', vis);

    if (this.roiLabel != null) {
        this.roiLabel.updateVisibility();
    }
}

ROICircle.prototype.updateContent = function (contentStr) {
    if(this.roiLabel != null) {
        this.roiLabel.updateContent(contentStr);
    }
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

ROICircle.prototype.release = function () {
    let data = d3.select(this.svg).selectAll('circle').data();
    let newData = [];
    for (let i = 0; i < data.length; ++i) {
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
        .data(newData, function (d) {
            return d.key;
        }).exit().remove();
    
    if(this.roiLabel != null) {
        this.roiLabel.release();
    }
}

ROICircle.prototype.addAnnotationLabel = function(str) {
    if (this.roiLabel == null)
    {
        this.roiLabel = new AnnotationLabel(this.key, this.svg, this.roiMain, str);
    }
}

const SCALING = 1.5;
// which mark (circle) on which svg (svg) am I (key) annotating to?
function AnnotationLabel(key, svg, selectedCircle, contentStr) {
    this.svg = svg;
    this.annotationKey = key + '-annotation';
    this.src = selectedCircle;
    this.contextFontSize = 11;

    this.content = null;
    this.border = null;
    this.arrow = null;

    const src_x = parseFloat(this.src.attr('cx'));
    const src_y = parseFloat(this.src.attr('cy'));
    const src_r = parseFloat(this.src.attr('r'));
    
    // text drag listener
    dragListener = d3.drag()
        .on("start", function (d) {
            // it's important that we suppress the mouseover event on the node being dragged.
            // Otherwise it will absorb the mouseover event and the underlying node will not detect it d3.select(this).attr('pointer-events', 'none');
            d3.event.sourceEvent.stopPropagation();
            preloc = {
                x: d3.event.x,
                y: d3.event.y
            };
        })
        .on("drag", (function (d) {
            const delta = {
                x: d3.event.x - preloc.x,
                y: d3.event.y - preloc.y
            };
            // update the text location
            let newX = parseFloat(this.content.attr('x')) + delta.x;
            let newY = parseFloat(this.content.attr('y')) + delta.y;
            this.content.attr('x', newX).attr('y', newY);

            // update the line (aka the arrow) location
            const centerX = parseFloat(this.src.attr('cx'));
            const centerY = parseFloat(this.src.attr('cy'));
            const centerR = parseFloat(this.src.attr('r'));
            let mag = Math.sqrt((newX - centerX) * (newX - centerX) + (newY - centerY) * (newY - centerY));
            if(mag < 0.01) {
                mag = 0.001;
                newX = centerX + 0.01 * 0.707;
                newY = centerY - 0.01 * 0.707;
            }
            this.arrow.attr('x1', centerX + centerR * (newX - centerX) / mag)
                .attr('y1', centerY + centerR * (newY - centerY) / mag)
                .attr('x2', newX)
                .attr('y2', newY);
            this.content.selectAll('tspan').attr('x', function(){return this.parentNode.getAttribute('x');})
            
            // record previous location
            preloc.x = d3.event.x;
            preloc.y = d3.event.y;
        }).bind(this)).on("end", function (d) {
            preloc = null;
        });

    this.content = d3.select(this.svg)
        .selectAll('text')
        .data(
            [{ key: this.annotationKey, }],
            function (d) {
                return d? d.key : null;
            })
        .enter()
        .append('text')
        .attr('font-family', 'monospace')
        .attr('font-size', this.contextFontSize)
        // .attr('class', 'no-select-text')
        .attr('alignment-baseline', 'central')
        .attr('text-anchor', 'start')   
        .attr('x', src_x + SCALING * src_r)
        .attr('y', src_y - SCALING * src_r)
        .attr('fill', 'red')
        .attr('cursor', 'move')
        .call(dragListener);

    if (contentStr != null && contentStr != '') {
      const multiLine = contentStr.split('|');
      let contentArray = [];
      for (let i = 0; i < multiLine.length; ++i) {
        contentArray.push(multiLine[i]);
      }

      this.content.selectAll('tspan')
          .data(contentArray)
          .enter()
          .append('tspan')
          .attr('x', function(d){return this.parentNode.getAttribute('x');})
          .attr(
              'dy', function(d, i) { return (i>0) * this.parentNode.getAttribute('font-size'); })
          .text(function(d, i) { return d; })
    }

    this.arrow =
        d3.select(this.svg)
            .selectAll('line')
            .data([{key: this.annotationKey}], function(d) { return d? d.key : null; })
            .enter()
            .append('line')
            .attr('x1', src_x + 0.707 * src_r)
            .attr('y1', src_y - 0.707 * src_r)
            .attr('x2', this.content.attr('x'))
            .attr('y2', this.content.attr('y'))
            .attr('stroke-width', 2)
            .attr('stroke', 'red')
            .attr('stroke-dasharray', '5, 5');
    
    // let textWidth = this.content._groups[0][0].getComputedTextLength();
    // let textHeight = parseFloat(this.content.attr('font-size'));
    // this.border = d3.select(this.svg)
    // .selectAll('rect')
    // .data([{key: this.annotationKey}], function(d) { return d.key; })
    // .enter()
    // .append('rect')
    // .attr('x', src_x + 2*src_r)
    // .attr('y', src_y - 2*src_r-2)
    // .attr('width', textWidth)
    // .attr('height', textHeight+4)
    // .style('fill-opacity', 0.0)
    // .attr('stroke-width', 2)
    // .attr('stroke', '#dcdcdc');

    this.updateVisibility();
}



AnnotationLabel.prototype.release = function () {
    const key = this.annotationKey;
    d3.select(this.svg).selectAll('text').filter(function (d) {
        return d? d.key == key : false;
    }).selectAll('tspan').remove();

    d3.select(this.svg).selectAll('text').filter(function (d) {
        return d? d.key == key : false;
    }).remove();

    d3.select(this.svg).selectAll('line').filter(function (d) {
        return d ? d.key == key : false;
    }).remove();
}

AnnotationLabel.prototype.updateContent = function (contentStr) {
    if (contentStr == null || contentStr == '')
        return;

    const multiLine = contentStr.split('|');
    let contentArray = [];
    for (let i = 0; i < multiLine.length; ++i) {
        contentArray.push(multiLine[i]);
    }

    const key = this.annotationKey;
    d3.select(this.svg)
        .selectAll('text')
        .filter(function (d) {
            return d ? d.key == key : false;
        }).each(function (d) {
            statisticTxt = d3.select(this).selectAll('tspan');
            if (statisticTxt.empty()) {
                statisticTxt.data(contentArray)
                    .enter()
                    .append('tspan')
                    .attr('x', function (d) {
                        return this.parentNode.getAttribute('x');
                    })
                    .attr('dy', function (d, i) {
                        return (i > 0) * this.parentNode.getAttribute('font-size');
                    })
                    .text(function (d, i) {
                        return d;
                    });
            } else {
                statisticTxt.data(contentArray)
                    .text(function (d, i) {
                        return d;
                    });
            }
        });
}

AnnotationLabel.prototype.resetLayout = function () {
    if (this.content.style('display') != 'none') {
        let newX = parseFloat(this.content.attr('x'));
        let newY = parseFloat(this.content.attr('y'));

        const centerX = parseFloat(this.src.attr('cx'));
        const centerY = parseFloat(this.src.attr('cy'));
        const centerR = parseFloat(this.src.attr('r'));

        this.content
            .attr('x', centerX + SCALING * centerR)
            .attr('y', centerY - SCALING * centerR);
        
        this.arrow
            .attr('x1', centerX + 0.707 * centerR)
            .attr('y1', centerY - 0.707 * centerR)
            .attr('x2', this.content.attr('x'))
            .attr('y2', this.content.attr('y'));

        this.content.selectAll('tspan').attr('x', function () {
            return this.parentNode.getAttribute('x');
        });
        return 1;
        // console.log(this.annotationKey + '  resetting');
    }
    else{
        return 0;
    }
}

// change the location of arrow & text
AnnotationLabel.prototype.updateLayout = function () {
    if (this.content.style('display') != 'none') {
        // if just moved it is trivial
        // get the text location
        let newX = parseFloat(this.content.attr('x'));
        let newY = parseFloat(this.content.attr('y'));

        const centerX = parseFloat(this.src.attr('cx'));
        const centerY = parseFloat(this.src.attr('cy'));
        const centerR = parseFloat(this.src.attr('r'));

        // update the line (aka the arrow) location
        let mag = Math.sqrt(
            (newX - centerX) * (newX - centerX) +
            (newY - centerY) * (newY - centerY));
        if (mag < 0.01) { 
            // to avoid division by 0
            mag = 0.01;
            newX = centerX + 0.01 * 0.707;
            newY = centerY - 0.01 * 0.707;
        }
        if (mag < (SCALING * centerR)){
            // if text is inside the circle
            newX = centerX + SCALING * centerR * (newX - centerX) / mag;
            newY = centerY + SCALING * centerR * (newY - centerY) / mag;
            mag = SCALING * centerR;
            this.content.attr('x', newX).attr('y', newY);
        }
        this.arrow.attr('x1', centerX + centerR * (newX - centerX) / mag)
            .attr('y1', centerY + centerR * (newY - centerY) / mag)
            .attr('x2', newX)
            .attr('y2', newY);

        this.content.selectAll('tspan').attr(
            'x',
            function () {
                return this.parentNode.getAttribute('x');
            });
    }
}

AnnotationLabel.prototype.updateVisibility = function () {
    this.arrow.style('display', this.src.style('display'));
    this.content.style('display', this.src.style('display'));
    this.resetLayout();
}