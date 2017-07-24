#ifndef MED_IMAGING_VOI_ERASER_H_
#define MED_IMAGING_VOI_ERASER_H_

#include "mi_graphic_item_voi.h"

// left-button changes circle size
class GraphicsCircleItem : public GraphicsSphereItem
{
public:
    GraphicsCircleItem(QGraphicsItem *parent = 0 , QGraphicsScene *scene = 0);
    virtual ~GraphicsCircleItem();

protected:
    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
};

#endif