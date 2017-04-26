#include "my_graphic_items.h"

#include <iostream>
#include <QPen>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneDragDropEvent>

MyTextItem::MyTextItem(QGraphicsItem *parent /*= 0 */, QGraphicsScene *scene /*= 0*/)
{
    this->setFlag(QGraphicsItem::ItemIsMovable);
    //this->setFlag(QGraphicsItem::ItemIsSelectable);
}

MyTextItem::~MyTextItem()
{

}

MyEllipse::MyEllipse(QGraphicsItem *parent /*= 0 */, QGraphicsScene *scene /*= 0*/)
{
    this->setFlag(QGraphicsItem::ItemIsMovable);
    //this->setFlag(QGraphicsItem::ItemIsSelectable);
    this->setPen(QPen(QColor(255,0,0)));
}

MyEllipse::~MyEllipse()
{
}



void MyEllipse::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->button() == Qt::LeftButton)
    {
        
        QGraphicsEllipseItem::mouseReleaseEvent(event);

        update_center_i();
    }
    else if (event->button() == Qt::RightButton)
    {
        update_center_i();
    }
}

void MyEllipse::mouseMoveEvent( QGraphicsSceneMouseEvent *event )
{
    if (event->buttons() == Qt::LeftButton)
    {
        QGraphicsEllipseItem::mouseMoveEvent(event);
        update_center_i();
    }
    else if (event->buttons() == Qt::RightButton)
    {
        std::cout << "current mouse : " << this->mapToScene(event->pos()).x() << " " << this->mapToScene(event->pos()).y() << std::endl;
        float x = abs(this->mapToScene(event->pos()).x() - _pre_center.x());
        float y = abs(_pre_center.y() - this->mapToScene(event->pos()).y());
        float radius = sqrt(x*x + y*y);

        QRectF rect = QRectF(0, 0, 2*radius ,2*radius);
        QPointF new_pos = _pre_center - QPointF(radius,radius);

        this->setRect( rect);
        this->setPos(new_pos);

        update_center_i();
    }
}

void MyEllipse::mouseReleaseEvent( QGraphicsSceneMouseEvent *event )
{
    if (event->button() == Qt::LeftButton)
    {
        QGraphicsEllipseItem::mouseReleaseEvent(event);
    }
    else if (event->button() == Qt::RightButton)
    {

    }
}

void MyEllipse::dragEnterEvent(QGraphicsSceneDragDropEvent *event)
{
    if (event->buttons() == Qt::LeftButton)
    {
        QGraphicsEllipseItem::dragEnterEvent(event);
    }
    else if (event->buttons() == Qt::RightButton)
    {

    }
}

void MyEllipse::dragLeaveEvent(QGraphicsSceneDragDropEvent *event)
{
    if (event->buttons() == Qt::LeftButton)
    {
        QGraphicsEllipseItem::dragLeaveEvent(event);
    }
    else if (event->buttons() == Qt::RightButton)
    {

    }
}

void MyEllipse::dragMoveEvent(QGraphicsSceneDragDropEvent *event)
{
    if (event->buttons() == Qt::LeftButton)
    {

    }
    else if (event->buttons() == Qt::RightButton)
    {

    }
}

void MyEllipse::update_center_i()
{
    _pre_center = QPointF(this->pos().x() + this->rect().width()*0.5f , this->pos().y() + this->rect().height()*0.5f);
    std::cout  << "center : " << _pre_center.x()  << " , " << _pre_center.y() << std::endl;
}
