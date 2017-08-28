#ifndef MED_IMG_PAINTER_H_
#define MED_IMG_PAINTER_H_

#include "qtpackage/mi_qt_package_export.h"
#include "renderalgo/mi_scene_base.h"
#include <QGraphicsItem>

class QPainter;
class QGraphicsItem;
MED_IMG_BEGIN_NAMESPACE 

class QtPackage_Export GraphicItemBase
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

MED_IMG_END_NAMESPACE

#endif