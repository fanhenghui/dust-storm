#ifndef MED_IMAGING_VOI_PAINTER_H_
#define MED_IMAGING_VOI_PAINTER_H_


#include "MedImgQtWidgets/mi_painter_base.h"
#include "MedImgIO/mi_voi.h"
#include <list>

MED_IMAGING_BEGIN_NAMESPACE
class VOIModel;
class QtWidgets_Export VOIPainter : public PainterBase
{
public:
    VOIPainter();
    virtual ~VOIPainter();
    void SetVOIModel(std::shared_ptr<VOIModel> pModel);
    virtual void Render();
protected:
private:
    std::shared_ptr<VOIModel> m_pModel;
};

MED_IMAGING_END_NAMESPACE

#endif