#ifndef MY_GRAPHIC_SCENE_H
#define MY_GRAPHIC_SCENE_H

#include <QGraphicsScene>

class MyGraphicScene : public QGraphicsScene
{
public:
    MyGraphicScene(QObject* parent = 0);
    virtual ~MyGraphicScene();

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *event);
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);

private:
};

#endif