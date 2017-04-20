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
    void SetCrossHairModel(std::shared_ptr<CrosshairModel> pModel);
    virtual void Render();
protected:
private:
    std::shared_ptr<CrosshairModel> m_pModel;
};

MED_IMAGING_END_NAMESPACE

#endif
