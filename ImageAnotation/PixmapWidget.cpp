/**
* The Image Annotation Tool for image annotations with pixelwise masks
*
* Copyright (C) 2007 Alexander Klaeser
*
* http://lear.inrialpes.fr/people/klaeser/
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/
#include "PixmapWidget.h"

#include <iostream>
#include <math.h>
#include <time.h>

#include "defines.h"
#include <QPixmap>
#include <QPainter>
#include <QWheelEvent>
#include <QPointF>
#include <QAbstractScrollArea>
#include <QScrollBar>
#include <QColor>
#include <QVector>
#include <QtDebug>

namespace
{
    int round(double number)
    {
        return (number > 0.0) ? floor(number + 0.5) : ceil(number - 0.5);
    }
}


PixmapWidget::PixmapWidget( QAbstractScrollArea *parentScrollArea, QWidget *parent )
    : QGLWidget( parent)
{
    _scroll_area = parentScrollArea;
    _pixmap = new QPixmap();
    _zoom_factor = 1.0;
    _pen_width = 5;
    _mask_transparency = 0.5;
    _is_drawing = false;
    _enable_painting = false;
    _is_confident = true;
    _is_erasing = false;


    setAttribute(Qt::WA_OpaquePaintEvent);
    setFocusPolicy(Qt::NoFocus);
    setMinimumSize( _pixmap->width()*_zoom_factor, _pixmap->height()*_zoom_factor );
    setMouseTracking(true);


    makeCurrent();

    setAutoBufferSwap( false );
    setAutoFillBackground( false );
}

PixmapWidget::~PixmapWidget()
{
    delete _pixmap;
}

void PixmapWidget::slot_zoom_factor_changed( double f )
{
    int w, h;

    if( fabs(f - _zoom_factor) <= 0.001 )
        return;

    _zoom_factor = f;
    emit( zoomFactorChanged( _zoom_factor ) );

    w = _pixmap->width()*_zoom_factor;
    h = _pixmap->height()*_zoom_factor;
    setMinimumSize( w, h );
    std::cout << w << " , " << h << std::endl;

    QWidget *p = dynamic_cast<QWidget*>( parent() );
    if( p )
    {
        std::cout << p->width()<< " " << p->height() << std::endl;
        resize( p->width(), p->height() );
    }

    update();
}

const QImage& PixmapWidget::get_draw_mask() const
{
    return _drawMask;
}

void PixmapWidget::set_pen_width(int width)
{
    _pen_width = width;
    update();
}

void PixmapWidget::set_mask(QImage& input_mask)
{
    // store the new mask
    QVector<uint> color_table = input_mask.colorTable();
    _drawMask = input_mask.convertToFormat(QImage::Format_ARGB32_Premultiplied);
    _drawMask.setColorTable(color_table);

    for (int y = 0; y < _drawMask.size().height(); ++y)
    {
        for (int x = 0; x < _drawMask.size().width(); ++x)
        {
            QRgb rgb = _drawMask.pixel(x, y);
            if (qRed(rgb) == 0 && qGreen(rgb) == 0 && qBlue(rgb) == 0)
            {
                _drawMask.setPixel(x, y, 0);
            }
        }
    }
    // we have to repaint
    repaint();
}


void PixmapWidget::set_mask_transparency(double transparency)
{
    std::cout << transparency << std::endl;
    //maskTransparency = transparency;
    //updateMask();

    //// create a partly transparent mask
    //QImage tmpMask = mask.copy();
    //for (int i = 0; i < _drawMask.colorCount(); i++) {
    //    tmpMask.setColor(i, _drawMask.color(i));
    //}
    //_drawMask = tmpMask.convertToFormat(QImage::Format_ARGB32_Premultiplied);


    //// we have to repaint
    //update();
}

void PixmapWidget::set_pixmap( const QPixmap& pixmap)
{
    delete _pixmap;
    _pixmap = new QPixmap(pixmap);

    emit( pixmapChanged( _pixmap ) );

    setMinimumSize( _pixmap->width()*_zoom_factor, _pixmap->height()*_zoom_factor );
    repaint();
}

void PixmapWidget::paintEvent( QPaintEvent *event )
{
    makeCurrent();

    //glClearColor(1.0,0.0,0.0,1.0);
    //glClear(GL_COLOR_BUFFER_BIT);


    bool drawBorder = false;
    int xOffset = 0, yOffset = 0;

    if( width() > _pixmap->width()*_zoom_factor ) {
        xOffset = (width()-_pixmap->width()*_zoom_factor)/2;
        drawBorder = true;
    }

    if( height() > _pixmap->height()*_zoom_factor ) {
        yOffset = (height()-_pixmap->height()*_zoom_factor)/2;
        drawBorder = true;
    }

    // get the current value of the parent scroll area .. to optimize the painting
    double hValue = 0, hMin = 0, hMax = 0, hPageStep = 0, hLength = 0;
    double vValue = 0, vMin = 0, vMax = 0, vPageStep = 0, vLength = 0;
    if (_scroll_area) 
    {
        QScrollBar *scrollBar;
        scrollBar = _scroll_area->horizontalScrollBar();
        if (scrollBar) 
        {
            hValue = scrollBar->value();
            hMin = scrollBar->minimum();
            hMax = scrollBar->maximum();
            hPageStep = scrollBar->pageStep();
            hLength = hMax - hMin + hPageStep;
        }
        scrollBar = _scroll_area->verticalScrollBar();
        if (scrollBar) 
        {
            vValue = scrollBar->value();
            vMin = scrollBar->minimum();
            vMax = scrollBar->maximum();
            vPageStep = scrollBar->pageStep();
            vLength = vMax - vMin + vPageStep;
        }
    }

    //
    // draw image and the transparent image mask
    //

    QPainter p(this);
    p.setRenderHint(QPainter::SmoothPixmapTransform, false);
    p.setRenderHint(QPainter::Antialiasing, false);

    // adjust the coordinate system
    p.save();
    p.translate(xOffset, yOffset);
    p.scale(_zoom_factor, _zoom_factor);
    _current_matrix_inv = p.matrix().inverted();
    _current_matrix = p.matrix();

    // find out which part of the image we have to draw
    // since we are embedded into a QScrollArea and not all is visible
    QRectF updateRectF = _current_matrix_inv.mapRect(event->rect());
    if (_last_v_scroll_value != vValue || _last_h_scroll_value != hValue || updateRectF.isEmpty()) {
        updateRectF.setLeft((hValue / hLength) * _pixmap->width());
        updateRectF.setWidth((hPageStep / hLength) * _pixmap->width());
        updateRectF.setTop((vValue / vLength) * _pixmap->height());
        updateRectF.setHeight((vPageStep / vLength) * _pixmap->height());
    }
    QRect updateRect;
    updateRect.setLeft(round(updateRectF.left()) - 1);
    updateRect.setRight(round(updateRectF.right()) + 1);
    updateRect.setTop(round(updateRectF.top()) - 1);
    updateRect.setBottom(round(updateRectF.bottom()) + 1);

    // save the scroll bar values
    _last_v_scroll_value = vValue;
    _last_h_scroll_value = hValue;

    // find out what needs to be cleared on the canvas outside the image
    QRectF eraseTop(updateRect.left(), updateRect.top(), updateRect.width(), -updateRect.top());
    QRectF eraseBottom(updateRect.left(), _pixmap->height(), updateRect.width(), updateRect.height() - (-updateRect.top() + _pixmap->height()));
    QRectF eraseLeft(updateRect.left(), 0, -updateRect.left(), _pixmap->height());
    QRectF eraseRight(_pixmap->width(), 0, updateRect.height() - (-updateRect.left() + _pixmap->width()), _pixmap->height());
    if (eraseTop.isValid())
        p.eraseRect(eraseTop);
    if (eraseBottom.isValid())
        p.eraseRect(eraseBottom);
    if (eraseLeft.isValid())
        p.eraseRect(eraseLeft);
    if (eraseRight.isValid())
        p.eraseRect(eraseRight);

    //std::cout<< "( "<<updateRect.left() << " , " << updateRect.right() << " ) " << " , ( "<<updateRect.bottom() << " , " << updateRect.top()<< " ) \n" ;

    // draw the image
    p.drawPixmap(updateRect.topLeft(), *_pixmap, updateRect);

    // draw the mask
    p.drawImage(updateRect.topLeft(), _drawMask, updateRect);

    //TODO using cursor to replace bruch itself
    // draw the brush
    QPen penWhite(Qt::lightGray);
    penWhite.setWidth(1 / _zoom_factor);
    QPen penBlack(Qt::darkGray);
    penBlack.setWidth(1 / _zoom_factor);
    p.setPen(penWhite);
    p.setRenderHint(QPainter::Antialiasing, true);
    p.drawEllipse(QRectF(
        xyMouseFollowed.x() - 0.5 * _pen_width - 0.5 / _zoom_factor + 0.5,
        xyMouseFollowed.y() - 0.5 * _pen_width - 0.5 / _zoom_factor + 0.5,
        _pen_width + 1 / _zoom_factor, _pen_width + 1 / _zoom_factor));
    p.setPen(penBlack);
    p.drawEllipse(QRectF(
        xyMouseFollowed.x() - 0.5 * _pen_width + 0.5 / _zoom_factor + 0.5,
        xyMouseFollowed.y() - 0.5 * _pen_width + 0.5 / _zoom_factor + 0.5,
        _pen_width - 1 / _zoom_factor, _pen_width - 1 / _zoom_factor));

    // draw a border around the image
    p.restore();
    //if (drawBorder) {
    //	p.setPen( Qt::black );
    //	p.drawRect( xOffset-1, yOffset-1, m_pm->width()*zoomFactor+1, m_pm->height()*zoomFactor+1 );
    //}

    swapBuffers();
}

void PixmapWidget::mousePressEvent(QMouseEvent * event)
{
    if (!_enable_painting)
    {
        return;
    }

    // get the mouse coordinate in the zoomed image
    QPoint xyMouseOrg(event->x(), event->y());
    QPoint xyMouse = _current_matrix_inv.map(xyMouseOrg);

    if (event->button() == Qt::LeftButton || event->button() == Qt::RightButton)
    {
        // get the region of the mouse cursor
        QRect updateRectOrg;
        int penOffset = (int) ceil(_zoom_factor * (0.5 * _pen_width + 2));
        updateRectOrg.setLeft(xyMouseOrg.x() - penOffset);
        updateRectOrg.setRight(xyMouseOrg.x() + penOffset);
        updateRectOrg.setTop(xyMouseOrg.y() - penOffset);
        updateRectOrg.setBottom(xyMouseOrg.y() + penOffset);

        QRect updateRect = _current_matrix_inv.mapRect(updateRectOrg);

        if (event->button() == Qt::LeftButton)
        {
            _is_erasing = false;
        }
        else 
        {
            _is_erasing = true;
        }

        // draw on the full image
        QPainter painter(&_drawMask);
        setup_current_painter_i(painter);
        painter.drawPoint(xyMouse);

        _is_drawing = true;

        // save the current position and perform an update in the
        // region of the mouse cusor
        lastXyMouseOrg = xyMouseOrg;
        lastXyMouse = xyMouse;
        lastXyDrawnMouseOrg = xyMouseOrg;
        lastXyDrawnMouse = xyMouse;
        update(updateRectOrg);
    }
}

void PixmapWidget::mouseMoveEvent(QMouseEvent * event)
{
    if (!_enable_painting)
    {
        return;
    }

    // save the mouse position in coordinates of the zoomed image
    QPoint xyMouseOrg(event->x(), event->y());
    QPoint xyMouse = _current_matrix_inv.map(xyMouseOrg);
    //	QPoint lastXyMouse = currentMatrixInv.map(lastXyMouseOrg);
    xyMouseFollowed = xyMouse;

    // determine the region that has been changed
    QRect updateRectOrg;
    updateRectOrg.setLeft(MIN(lastXyMouseOrg.x(), xyMouseOrg.x()) - (int) ceil(_zoom_factor * (0.5 * _pen_width + 2)));
    updateRectOrg.setRight(MAX(lastXyMouseOrg.x(), xyMouseOrg.x()) + (int) ceil(_zoom_factor * (0.5 * _pen_width + 2)));
    updateRectOrg.setTop(MIN(lastXyMouseOrg.y(), xyMouseOrg.y()) - (int) ceil(_zoom_factor * (0.5 * _pen_width + 2)));
    updateRectOrg.setBottom(MAX(lastXyMouseOrg.y(), xyMouseOrg.y()) + (int) ceil(_zoom_factor * (0.5 * _pen_width + 2)));
    QRect updateRect = _current_matrix_inv.mapRect(updateRectOrg);

    if (_is_drawing) 
    {
        QPainter painter(&_drawMask);
        setup_current_painter_i(painter);
        painter.drawLine(lastXyMouse, xyMouse);
    }

    // save the current position and perform an update
    lastXyMouseOrg = xyMouseOrg;
    lastXyMouse = _current_matrix_inv.map(xyMouseOrg);
    update(updateRectOrg);
}

void PixmapWidget::mouseReleaseEvent(QMouseEvent * event)
{
    if (!_enable_painting)
    {
        return;
    }

    // save the mouse position in coordinates of the zoomed image
    QPoint xyMouseOrg(event->x(), event->y());
    QPoint xyMouse = _current_matrix_inv.map(xyMouseOrg);
    QPoint lastXyMouse = _current_matrix_inv.map(lastXyMouseOrg);

    // determine the region that has been changed
    QRect updateRectOrg;
    updateRectOrg.setLeft(MIN(lastXyMouseOrg.x(), xyMouseOrg.x()) - (int) ceil(_zoom_factor * (0.5 * _pen_width + 2)));
    updateRectOrg.setRight(MAX(lastXyMouseOrg.x(), xyMouseOrg.x()) + (int) ceil(_zoom_factor * (0.5 * _pen_width + 2)));
    updateRectOrg.setTop(MIN(lastXyMouseOrg.y(), xyMouseOrg.y()) - (int) ceil(_zoom_factor * (0.5 * _pen_width + 2)));
    updateRectOrg.setBottom(MAX(lastXyMouseOrg.y(), xyMouseOrg.y()) + (int) ceil(_zoom_factor * (0.5 * _pen_width + 2)));
    QRect updateRect = _current_matrix_inv.mapRect(updateRectOrg);

    if (event->button() == Qt::LeftButton && _is_drawing) 
    {
        QPainter painter(&_drawMask);
        setup_current_painter_i(painter);
        painter.drawLine(lastXyMouse, xyMouse);
    }

    // save the last position
    lastXyMouseOrg = xyMouseOrg;
    lastXyMouse = _current_matrix_inv.map(xyMouseOrg);
    lastXyDrawnMouseOrg = xyMouseOrg;
    lastXyDrawnMouse = _current_matrix_inv.map(xyMouseOrg);

    // send the signal that the mask has been changed
    emit( maskChanged( &_drawMask ) );

    // update
    _is_drawing = false;
    _is_erasing = false;

    // paint update
    update(updateRectOrg);
}

void PixmapWidget::updateMouseCursor()
{
}

void PixmapWidget::updateMask()
{
    //// store the original color table and clear the original mask
    //QVector<QRgb> orgColorTable = mask.colorTable();
    //mask.fill(0);

    //// convert the mask to the original format
    //QRgb lastRgb = qRgb(0, 0, 0);
    //int  lastI = 0;
    //for (int y = 0; y < _drawMask.height(); y++)
    //    for (int x = 0; x < _drawMask.width(); x++) {
    //        QRgb rgb = _drawMask.pixel(x, y);

    //        // see whether we have to find for the right color index or not
    //        if (rgb == lastRgb)
    //            mask.setPixel(x, y, lastI);
    //        else {
    //            // find the closest color
    //            int bestI = -1;
    //            double tmpDist, bestDist;
    //            for (int j = 0; j < orgColorTable.size(); j++) {
    //                tmpDist = pow(double(qRed(rgb) - qRed(orgColorTable[j])), 2)
    //                    + pow(double(qGreen(rgb) - qGreen(orgColorTable[j])), 2)
    //                    + pow(double(qBlue(rgb) - qBlue(orgColorTable[j])), 2);
    //                if (bestI < 0 || tmpDist < bestDist) {
    //                    bestI = j;
    //                    bestDist = tmpDist;
    //                    if (bestDist == 0)
    //                        break;
    //                }
    //            }

    //            // set the color and buffer the values
    //            mask.setPixel(x, y, bestI);
    //            lastI = bestI;
    //            lastRgb = rgb;
    //        }
    //    }
}

void PixmapWidget::setup_current_painter_i(QPainter &painter)
{
    QColor rgba;
    if (_is_erasing)
    {
        rgba = QColor(_drawMask.color(BACKGROUND));
        rgba.setAlpha(0);
    }
    else
    {
        if (_is_confident)
        {
            rgba = QColor(_drawMask.color(CONFIDENCE_OBJECT));
        }
        else
        {
            rgba = QColor(_drawMask.color(UN_CONFIDENCE_OBJECT));
        }
        rgba.setAlpha(int(_mask_transparency*255));
    }

    painter.setRenderHint(QPainter::Antialiasing, false);
    painter.setPen(QPen(rgba, _pen_width, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    if (_is_erasing)
    {
        painter.setCompositionMode(QPainter::CompositionMode_Clear);
    }
    else
    {
        painter.setCompositionMode(QPainter::CompositionMode_SourceOver);
    }
}

void PixmapWidget::initializeGL()
{

}

void PixmapWidget::enable_painting(bool flag)
{
    _enable_painting = flag;

    if (flag)
    {
        setCursor(QCursor(Qt::CrossCursor));
    }
    else
    {
        setCursor(QCursor(Qt::ArrowCursor));
    }

    update();
}

void PixmapWidget::set_confidence(bool flag)
{
    _is_confident = flag;
}

//void PixmapWidget::resizeGL(int w, int h)
//{
//
//}
