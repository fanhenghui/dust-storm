#ifndef MED_IMAGING_PATIENT_MPR_BORDER_H_
#define MED_IMAGING_PATIENT_MPR_BORDER_H_

#include "MedImgQtWidgets/mi_graphic_item_base.h"

class QGraphicsLineItem;

MED_IMAGING_BEGIN_NAMESPACE

class CrosshairModel;
class QtWidgets_Export GraphicItemMPRBorder : public GraphicItemBase
{
public:
    GraphicItemMPRBorder();
    virtual ~GraphicItemMPRBorder();

    void set_crosshair_model(std::shared_ptr<CrosshairModel> model);

    virtual std::vector<QGraphicsItem*> get_init_items();

    virtual void update(std::vector<QGraphicsItem*>& to_be_add , std::vector<QGraphicsItem*>& to_be_remove);
protected:
private:
    std::shared_ptr<CrosshairModel> _model;
    QGraphicsLineItem* _lines[4];

    QColor _pre_color;
    int _pre_pen_width;
    int _pre_window_width;
    int _pre_window_height;
};

MED_IMAGING_END_NAMESPACE


#endif