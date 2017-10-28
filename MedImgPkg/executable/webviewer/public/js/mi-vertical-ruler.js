const MARGIN_RIGHT = 15;

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
    let power = 1.0;
    while (number >= power) {
        power *= 10.0;
    }
    power /= 10.0;
    let mod = Math.fmod(number, power);
    let r = number - mod;
    let divider = parseInt(Math.floor(r / power * 10.0 + 0.5));

    return {
        'length': r,
        'label': r + 'mm',
        'divider': divider
    };
}

VerticalRuler.prototype.updateRuler = function (frustumHeight) {
    const width = this.svg.getAttribute('width');
    const height = this.svg.getAttribute('height');

    let cx = width / 2.0;
    let cy = height / 2.0;
    let y1 = height / 4.0;
    let y2 = 3.0 * height / 4.0;

    let initialRulerLength = frustumHeight / 2.0; // in world space
    let rulerSetting = decimalFloor(initialRulerLength);
    let add = (y2 - y1) / initialRulerLength * rulerSetting.length;
    let y1new = cy - add / 2;
    let y2new = cy + add / 2;

    if ((y2 - y1) > 128) // at least 128 pixels 
    {
        let xunit = width / 100.0; // 1 percentage of the window width in pixel
        if (xunit > 8.0) {
            xunit = 8.0;
        }
        let smallAdd = 0;
        let bigAdd = 0;

        if (((y2new - y1new) * 10.0 / rulerSetting.divider) > 4) {
            smallAdd = 0;
            bigAdd = parseInt(Math.floor(xunit));
        }
        if (((y2new - y1new) * 1.0 / rulerSetting.divider) > 4) {
            smallAdd = parseInt(Math.floor(xunit));
            bigAdd = parseInt(Math.floor(xunit * 2.0));
        }

        let pathCommands = 'M ' + (width - MARGIN_RIGHT - xunit - bigAdd) + ' ' + y1new + ' ';
        pathCommands += 'L ' + (width - MARGIN_RIGHT) + ' ' + y1new + ' ';
        pathCommands += 'L ' + (width - MARGIN_RIGHT) + ' ' + y2new + ' ';
        pathCommands += 'L ' + (width - MARGIN_RIGHT - xunit - bigAdd) + ' ' + y2new + ' ';

        let yadd = add / rulerSetting.divider;

        let ypos = y1new;
        for (let i = 0; i < rulerSetting.divider; i += 10) {
            pathCommands += 'M ' + (width - bigAdd - MARGIN_RIGHT) + ' ' + ypos + ' ';
            pathCommands += 'L ' + (width - MARGIN_RIGHT) + ' ' + ypos + ' ';
            ypos += yadd * 10;
        }

        ypos = y1new;
        for (let i = 0; i < rulerSetting.divider; i++) {
            if (i % 10 != 0) {
                pathCommands += 'M ' + (width - smallAdd - MARGIN_RIGHT) + ' ' + ypos + ' ';
                pathCommands += 'L ' + (width - MARGIN_RIGHT) + ' ' + ypos + ' ';
            }
            ypos += yadd;
        }
        this.rulerPath.attr('d', pathCommands)
            .attr('visibility', 'visible');

        this.rulerLabel
            .attr('visibility', 'visible');

        this.rulerLabel.text(rulerSetting.label)
            .attr('x', function (d) {
                return width - MARGIN_RIGHT;
            })
            .attr('y', y2new+4);
    } else {
        this.rulerPath.attr('visibility', 'hidden');
        this.rulerLabel.attr('visibility', 'hidden');
    }
}