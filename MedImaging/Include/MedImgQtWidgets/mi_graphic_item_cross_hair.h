#ifndef MED_IMAGING_PAINTER_CROSS_HAIR_H
#define MED_IMAGING_PAINTER_CROSS_HAIR_H

#include "MedImgQtWidgets/mi_graphic_item_base.h"
#include "MedImgArithmetic/mi_line.h"
#include "MedImgArithmetic/mi_color_unit.h"

class QGraphicsLineItem;

MED_IMAGING_BEGIN_NAMESPACE

class CrosshairModel;
class QtWidgets_Export GraphicItemCrosshair : public GraphicItemBase
{
public:
    GraphicItemCrosshair();
    virtual ~GraphicItemCrosshair();

    void set_crosshair_model(std::shared_ptr<CrosshairModel> model);

    virtual std::vector<QGraphicsItem*> get_init_items();

    virtual void update(std::vector<QGraphicsItem*>& to_be_add , std::vector<QGraphicsItem*>& to_be_remove);

protected:
private:
    std::shared_ptr<CrosshairModel> _model;
    QGraphicsLineItem* _lines[2];

    Line2D _pre_lines[2];
    RGBUnit _pre_colors[2];
    
};

MED_IMAGING_END_NAMESPACE

#endif

