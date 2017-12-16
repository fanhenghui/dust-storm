#ifndef MED_IMAGING_PATIENT_CORNERS_INFO_H_
#define MED_IMAGING_PATIENT_CORNERS_INFO_H_

#include "qtpackage/mi_graphic_item_base.h"

class QGraphicsTextItem;

MED_IMG_BEGIN_NAMESPACE

class QtWidgets_Export GraphicItemCornersInfo : public GraphicItemBase
{
public:
    GraphicItemCornersInfo();
    virtual ~GraphicItemCornersInfo();

    virtual void set_scene(std::shared_ptr<SceneBase> scene);

    virtual std::vector<QGraphicsItem*> get_init_items();

    virtual void update(std::vector<QGraphicsItem*>& to_be_add , std::vector<QGraphicsItem*>& to_be_remove);

private:
    void refresh_text();

private:
    QGraphicsTextItem* _text_item_lb;
    QGraphicsTextItem* _text_item_lt;
    QGraphicsTextItem* _text_item_rb;
    QGraphicsTextItem* _text_item_rt;

    int _pre_ww;
    int _pre_wl;

    int _pre_window_width;
    int _pre_window_height;

    int _pre_slice;
};

MED_IMG_END_NAMESPACE


#endif