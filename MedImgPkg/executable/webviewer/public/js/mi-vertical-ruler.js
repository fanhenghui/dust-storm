const MarginRight = 15;

function VerticalRuler(svg, cellID) {
    this.svg = svg;
    this.cellID = cellID;

    this.g = d3.select(this.svg).append('g').attr('id', this.cellID + 'g');
    this.rulerLabel = this.g.append('text')
        .attr('font-family', 'monospace')
        .attr('font-size', '12px')
        .attr('fill', '#dcdcdc')
        .attr('class', 'no-select-text')
        .attr('alignment-baseline', 'hanging')
        .attr('text-anchor', 'end');

    this.rulerPath = this.g.append('path').attr('d', 'M 0 0')
        .attr('stroke', '#dcdcdc')
        .attr('stroke-width', 2)
        .attr('fill', 'none')
        .attr('visibility', 'hidden');
}

Math.fmod = function (a, b) {
    return Number((a - (Math.floor(a / b) * b)));
};

function decimalFloor(number) {
    var power = 1.0;
    while (number >= power) {
        power *= 10.0;
    }
    power /= 10.0;
    var mod = Math.fmod(number, power);
    var r = number - mod;
    var divider = parseInt(Math.floor(r / power * 10.0 + 0.5));

    return {
        'length': r,
        'label': r + 'mm',
        'divider': divider
    };
}

VerticalRuler.prototype.updateRuler = function (frustumHeight) {
    var width = this.svg.getAttribute('width');
    var height = this.svg.getAttribute('height');

    var cx = width / 2.0;
    var cy = height / 2.0;
    var y1 = height / 4.0;
    var y2 = 3.0 * height / 4.0;

    var initialRulerLength = frustumHeight / 2.0; // in world space
    var rulerSetting = decimalFloor(initialRulerLength);
    var add = (y2 - y1) / initialRulerLength * rulerSetting.length;
    var y1new = cy - add / 2;
    var y2new = cy + add / 2;

    if ((y2 - y1) > 128) // at least 128 pixels 
    {
        var xunit = width / 100.0; // 1 percentage of the window width in pixel
        if (xunit > 8.0) {
            xunit = 8.0;
        }
        var smallAdd = 0;
        var bigAdd = 0;

        if (((y2new - y1new) * 10.0 / rulerSetting.divider) > 4) {
            smallAdd = 0;
            bigAdd = parseInt(Math.floor(xunit));
        }
        if (((y2new - y1new) * 1.0 / rulerSetting.divider) > 4) {
            smallAdd = parseInt(Math.floor(xunit));
            bigAdd = parseInt(Math.floor(xunit * 2.0));
        }

        var pathCommands = 'M ' + (width - MarginRight - xunit - bigAdd) + ' ' + y1new + ' ';
        pathCommands += 'L ' + (width - MarginRight) + ' ' + y1new + ' ';
        pathCommands += 'L ' + (width - MarginRight) + ' ' + y2new + ' ';
        pathCommands += 'L ' + (width - MarginRight - xunit - bigAdd) + ' ' + y2new + ' ';

        var yadd = add / rulerSetting.divider;

        var ypos = y1new;
        for (var i = 0; i < rulerSetting.divider; i += 10) {
            pathCommands += 'M ' + (width - bigAdd - MarginRight) + ' ' + ypos + ' ';
            pathCommands += 'L ' + (width - MarginRight) + ' ' + ypos + ' ';
            ypos += yadd * 10;
        }

        ypos = y1new;
        for (var i = 0; i < rulerSetting.divider; i++) {
            if (i % 10 != 0) {
                pathCommands += 'M ' + (width - smallAdd - MarginRight) + ' ' + ypos + ' ';
                pathCommands += 'L ' + (width - MarginRight) + ' ' + ypos + ' ';
            }
            ypos += yadd;
        }
        this.rulerPath.attr('d', pathCommands)
            .attr('visibility', 'visible');

        this.rulerLabel
            .attr('visibility', 'visible');

        this.rulerLabel.text(rulerSetting.label)
            .attr('x', function (d) {
                return width - MarginRight;
            })
            .attr('y', y2new+4);
    } else {
        this.rulerPath.attr('visibility', 'hidden');
        this.rulerLabel.attr('visibility', 'hidden');
    }
}