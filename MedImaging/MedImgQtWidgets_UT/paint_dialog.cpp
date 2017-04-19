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
    ////�½�QPainter�������pix�Ͻ��л�ͼ
    //QPainter pp(&pix);
    ////��pix�ϵģ�0��0����ͣ�50��50����֮�����ֱ��
    //pp.drawLine(0, 0, 50, 50);
    //painter.drawPixmap(100, 100, pix);

    //QPixmap pix2(200, 200);
    //pix2.fill(Qt::transparent);

    //QPainter pp2(&pix2);
    ////��pix�ϵģ�0��0����ͣ�50��50����֮�����ֱ��
    //pp2.drawLine(0, 0, 30, 50);


    //painter.drawPixmap(100, 100, pix2);



    QPainter painter(this);
    QPixmap pix(this->width(), this->height());
    pix.fill(Qt::transparent);
    //pix.fill(Qt::red);
    //�½�QPainter�������pix�Ͻ��л�ͼ
    QPainter pp(&pix);
    //��pix�ϵģ�0��0����ͣ�50��50����֮�����ֱ��
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
