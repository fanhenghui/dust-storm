#ifndef MY_GRAPHIC_SCENE_H
#define MY_GRAPHIC_SCENE_H

#include <QGraphicsScene>

class MyGraphicScene : public QGraphicsScene
{
public:
    MyGraphicScene(QObject* parent = 0);
    virtual ~MyGraphicScene();
    bool no_item_selected();

protected:
    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event);
    virtual void wheelEvent(QGraphicsSceneWheelEvent *event);

private:
    bool _no_item_selected;
};

#endif