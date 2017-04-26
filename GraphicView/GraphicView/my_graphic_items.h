#ifndef MY_GRAPHIC_ITEMS_H
#define MY_GRAPHIC_ITEMS_H

#include <QGraphicsTextItem>

class MyTextItem : public QGraphicsTextItem
{
public:
    MyTextItem(QGraphicsItem *parent = 0 , QGraphicsScene *scene = 0);
    virtual ~MyTextItem();

protected:
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);  
private:
};

class MyEllipse : public QGraphicsEllipseItem
{
public:
    MyEllipse(QGraphicsItem *parent = 0 , QGraphicsScene *scene = 0);
    virtual ~MyEllipse();

    void set_sphere(const QPointF& center , float radius);


protected:
    virtual void dragEnterEvent(QGraphicsSceneDragDropEvent *event);
    virtual void dragLeaveEvent(QGraphicsSceneDragDropEvent *event);
    virtual void dragMoveEvent(QGraphicsSceneDragDropEvent *event);
    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
private:
    void update_center_i();
    //QPoint _pre_point;
    QPointF _pre_center;
};

#endif