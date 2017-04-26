#ifndef MED_IMAGING_PAINTER_H_
#define MED_IMAGING_PAINTER_H_

#include "MedImgQtWidgets/mi_qt_widgets_stdafx.h"
#include "MedImgRenderAlgorithm/mi_scene_base.h"
#include <QGraphicsItem>

class QPainter;
class QGraphicsItem;
MED_IMAGING_BEGIN_NAMESPACE

class QtWidgets_Export GraphicItemBase
{
public:
    GraphicItemBase()
    {};

    virtual ~GraphicItemBase() {};

    virtual void set_scene(std::shared_ptr<SceneBase> scene)
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(scene);
        _scene = scene;
    }
    //When first add to container , call this to get initialized items
    virtual std::vector<QGraphicsItem*> get_init_items() = 0;

    //Render each frame should call this to update items
    virtual void update(std::vector<QGraphicsItem*>& to_be_add , std::vector<QGraphicsItem*>& to_be_remove) = 0;

    //Do this logic after call update(such as delete useless graphics items) , optional
    virtual void post_update() {}

protected:
    std::shared_ptr<SceneBase> _scene;
};

MED_IMAGING_END_NAMESPACE

#endif