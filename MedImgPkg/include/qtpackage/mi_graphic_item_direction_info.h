#ifndef MED_IMAGING_PATIENT_DIRECTION_INFO_H_
#define MED_IMAGING_PATIENT_DIRECTION_INFO_H_

#include "qtpackage/mi_graphic_item_base.h"
#include "arithmetic/mi_vector3.h"

class QGraphicsTextItem;

MED_IMG_BEGIN_NAMESPACE

class QtWidgets_Export GraphicItemDirectionInfo : public GraphicItemBase
{
public:
    GraphicItemDirectionInfo();
    virtual ~GraphicItemDirectionInfo();

    virtual void set_scene(std::shared_ptr<SceneBase> scene);

    virtual std::vector<QGraphicsItem*> get_init_items();

    virtual void update(std::vector<QGraphicsItem*>& to_be_add , std::vector<QGraphicsItem*>& to_be_remove);

private:
    void refresh_text();

private:
    QGraphicsTextItem* _text_item_left;
    QGraphicsTextItem* _text_item_right;
    QGraphicsTextItem* _text_item_top;
    QGraphicsTextItem* _text_item_bottom;

    int _pre_window_width;
    int _pre_window_height;

    Vector3 _slice_left; // normal
    Vector3 _slice_up; // up
};

MED_IMG_END_NAMESPACE

#endif