#ifndef MED_IMG_PATIENT_MPR_BORDER_H_
#define MED_IMG_PATIENT_MPR_BORDER_H_

#include "qtpackage/mi_graphic_item_base.h"

class QGraphicsLineItem;

MED_IMG_BEGIN_NAMESPACE 

class CrosshairModel;
class FocusModel;
class QtPackage_Export GraphicItemMPRBorder : public GraphicItemBase
{
public:
    GraphicItemMPRBorder();
    virtual ~GraphicItemMPRBorder();

    void set_crosshair_model(std::shared_ptr<CrosshairModel> model);

    void set_focus_model(std::shared_ptr<FocusModel> model);

    virtual std::vector<QGraphicsItem*> get_init_items();

    virtual void update(std::vector<QGraphicsItem*>& to_be_add , std::vector<QGraphicsItem*>& to_be_remove);
protected:
private:
    std::shared_ptr<CrosshairModel> _model_corsshair;
    std::shared_ptr<FocusModel> _model_focus;

    QGraphicsLineItem* _lines[4];

    QColor _pre_color;
    int _pre_pen_width;
    int _pre_window_width;
    int _pre_window_height;
};

MED_IMG_END_NAMESPACE


#endif