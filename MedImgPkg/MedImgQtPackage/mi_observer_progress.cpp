#include "mi_observer_progress.h"

#include "MedImgUtil/mi_model_progress.h"

#include <QProgressDialog>

MED_IMG_BEGIN_NAMESPACE

ProgressObserver::ProgressObserver():_progress_dialog (nullptr)
{

}

ProgressObserver::~ProgressObserver()
{

}

void ProgressObserver::update(int )
{
    if (_progress_dialog)
    {
        std::shared_ptr<ProgressModel> model = _model.lock();
        if (model)
        {
            _progress_dialog->setValue(model->get_progress());
        }
    }
}

void ProgressObserver::set_progress_model( std::shared_ptr<ProgressModel> model )
{
    _model = model;
}

void ProgressObserver::set_progress_dialog( QProgressDialog* dialog )
{
    _progress_dialog = dialog;
}

MED_IMG_END_NAMESPACE