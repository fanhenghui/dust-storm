#ifndef MED_IMAGING_PAINTER_CROSS_HAIR_H
#define MED_IMAGING_PAINTER_CROSS_HAIR_H

#include "MedImgQtWidgets/mi_painter_base.h"

MED_IMAGING_BEGIN_NAMESPACE

class CrosshairModel;
class QtWidgets_Export CrosshairPainter : public PainterBase
{
public:
    CrosshairPainter();
    virtual ~CrosshairPainter();

    void set_crosshair_model(std::shared_ptr<CrosshairModel> model);

    virtual void render();
protected:
private:
    std::shared_ptr<CrosshairModel> _model;
};

MED_IMAGING_END_NAMESPACE

#endif

