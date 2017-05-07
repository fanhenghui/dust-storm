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
#ifndef PIXMAPWIDGET_H
#define PIXMAPWIDGET_H

#include <QWidget>
#include <QString>
#include <QPixmap>
#include <QImage>
#include <QList>
#include <QRect>
#include <QMouseEvent>
#include <QMatrix>

#define MARGIN 5


class QAbstractScrollArea;

// defines a bounding box that is to be drawn on the screen
class BoundingBox
{
public:
	QRectF box;
	QList<QPointF> fixPoints;
	double score;
};


// our own pixmap widget .. which displays an image and a annotation mask
class PixmapWidget : public QWidget
{
	Q_OBJECT

private:
	QPixmap *m_pm;
	QImage mask, drawMask, tmpDrawMask, drawOnColorMask;
	double zoomFactor;
	double maskTransparency;
	int iMaskEditColor, iMaskDrawOnColor;
	int penWidth;
	QMatrix currentMatrixInv;
	QMatrix currentMatrix;
	QPoint lastXyMouseOrg;
	QPoint lastXyMouse;
	QPoint lastXyDrawnMouseOrg;
	QPoint lastXyDrawnMouse;
	QPoint xyMouseFollowed;
	bool isDrawing, isFloodFilling;
	double lastVScrollValue, lastHScrollValue;

	QAbstractScrollArea *scrollArea;

public:
	PixmapWidget(QAbstractScrollArea*, QWidget *parent=0);
	~PixmapWidget();
	int getMaskEditColor();
	QImage* getMask();

public slots:
	void setZoomFactor(double);
	void setPixmap(const QPixmap&);
	void setMask(QImage&);
	void setMaskEditColor(int iColor);
	void setMaskDrawOnColor(int iColor);
	void setPenWidth(int width);
	void setMaskTransparency(double transparency);
	void setFloodFill(bool flag);

signals:
	void zoomFactorChanged(double);
	void pixmapChanged(QPixmap*);
 	void drawEvent(QImage*);

protected:
	void paintEvent(QPaintEvent*);
//	void wheelEvent(QWheelEvent*);
	void mouseMoveEvent(QMouseEvent * event);
	void mousePressEvent(QMouseEvent * event);
	void mouseReleaseEvent(QMouseEvent * event);
//	void keyPressEvent(QKeyEvent * event);
//	void keyReleaseEvent(QKeyEvent * event);

private:
	void updateMouseCursor();
	void updateMask();
	QRgb getColor(int i);
	void setUpPainter(QPainter &painter);
};

#endif // PIXMAPWIDGET_H
