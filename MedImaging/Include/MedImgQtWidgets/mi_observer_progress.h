#ifndef MED_IMAGING_OBSERVER_PROGRESS_H_
#define MED_IMAGING_OBSERVER_PROGRESS_H_

#include "MedImgQtWidgets/mi_qt_widgets_stdafx.h"
#include "MedImgCommon/mi_observer_interface.h"

class QProgressDialog;

MED_IMAGING_BEGIN_NAMESPACE

class ProgressModel;
class QtWidgets_Export ProgressObserver : public IObserver
{
public:
    ProgressObserver();
    virtual ~ProgressObserver();

    virtual void update();

    void set_progress_model(std::shared_ptr<ProgressModel> model);

    void set_progress_dialog(QProgressDialog* dialog);

protected:
private:
    std::weak_ptr<ProgressModel> _model;
    QProgressDialog* _progress_dialog;
};

MED_IMAGING_END_NAMESPACE

#endif
