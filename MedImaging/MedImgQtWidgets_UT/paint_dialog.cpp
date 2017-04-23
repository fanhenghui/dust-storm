#include "paint_dialog.h"
#include <QPainter>
#include <QMouseEvent>

PaintDialog::PaintDialog( QDialog* parent /*= 0*/ )
{

}

PaintDialog::~PaintDialog()
{

}

void PaintDialog::paintEvent( QPaintEvent * )
{
    //QPainter painter(this);
    //QPixmap pix(200, 200);
    //pix.fill(Qt::transparent);
    ////pix.fill(Qt::red);
    ////新建QPainter类对象，在pix上进行绘图
    //QPainter pp(&pix);
    ////在pix上的（0，0）点和（50，50）点之间绘制直线
    //pp.drawLine(0, 0, 50, 50);
    //painter.drawPixmap(100, 100, pix);

    //QPixmap pix2(200, 200);
    //pix2.fill(Qt::transparent);

    //QPainter pp2(&pix2);
    ////在pix上的（0，0）点和（50，50）点之间绘制直线
    //pp2.drawLine(0, 0, 30, 50);


    //painter.drawPixmap(100, 100, pix2);



    QPainter painter(this);
    QPixmap pix(this->width(), this->height());
    pix.fill(Qt::transparent);
    //pix.fill(Qt::red);
    //新建QPainter类对象，在pix上进行绘图
    QPainter pp(&pix);
    //在pix上的（0，0）点和（50，50）点之间绘制直线
    double x = _cur_point.x() - _pre_point.x();
    double y = _cur_point.y() - _pre_point.y();
    double radius = sqrt(x*x + y*y);
    pp.drawEllipse(_pre_point , int(radius) , int(radius) );
    painter.drawPixmap(0, 0, pix);
}

void PaintDialog::mouseMoveEvent( QMouseEvent * event)
{
    _cur_point = event->pos(); 
    this->update();
}

void PaintDialog::mousePressEvent( QMouseEvent * event)
{
    _pre_point = event->pos();
    _cur_point = event->pos(); 
    this->update();
}

void PaintDialog::mouseReleaseEvent( QMouseEvent * event)
{
    _cur_point = event->pos(); 
    this->update();
}
