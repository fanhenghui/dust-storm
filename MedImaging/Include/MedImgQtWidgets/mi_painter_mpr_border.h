#ifndef MED_IMAGING_PATIENT_MPR_BORDER_H_
#define MED_IMAGING_PATIENT_MPR_BORDER_H_

#include "MedImgQtWidgets/mi_painter_base.h"

MED_IMAGING_BEGIN_NAMESPACE

class CrosshairModel;
class QtWidgets_Export MPRBorderPainter : public PainterBase
{
public:
    MPRBorderPainter();
    virtual ~MPRBorderPainter();
    void SetCrossHairModel(std::shared_ptr<CrosshairModel> pModel);
    virtual void Render();
protected:
private:
    std::shared_ptr<CrosshairModel> m_pModel;
};

MED_IMAGING_END_NAMESPACE


#endif