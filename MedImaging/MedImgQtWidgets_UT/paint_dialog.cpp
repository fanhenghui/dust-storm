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
    double x = m_ptCur.x() - m_ptPre.x();
    double y = m_ptCur.y() - m_ptPre.y();
    double dRadius = sqrt(x*x + y*y);
    pp.drawEllipse(m_ptPre , int(dRadius) , int(dRadius) );
    painter.drawPixmap(0, 0, pix);
}

void PaintDialog::mouseMoveEvent( QMouseEvent * event)
{
    m_ptCur = event->pos(); 
    this->update();
}

void PaintDialog::mousePressEvent( QMouseEvent * event)
{
    m_ptPre = event->pos();
    m_ptCur = event->pos(); 
    this->update();
}

void PaintDialog::mouseReleaseEvent( QMouseEvent * event)
{
    m_ptCur = event->pos(); 
    this->update();
}
