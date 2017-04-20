#ifndef MED_IMAGING_PATIENT_CORNERS_INFO_H_
#define MED_IMAGING_PATIENT_CORNERS_INFO_H_

#include "MedImgQtWidgets/mi_painter_base.h"

MED_IMAGING_BEGIN_NAMESPACE

class QtWidgets_Export CornersInfoPainter : public PainterBase
{
public:
    CornersInfoPainter();
    virtual ~CornersInfoPainter();
    virtual void render();
protected:
private:
};

MED_IMAGING_END_NAMESPACE


#endif