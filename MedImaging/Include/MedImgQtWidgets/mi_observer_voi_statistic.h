#ifndef MED_IMAGING_OBSERVER_VOI_STATISTIC_H
#define MED_IMAGING_OBSERVER_VOI_STATISTIC_H

#include "MedImgQtWidgets/mi_qt_widgets_stdafx.h"
#include "MedImgCommon/mi_observer_interface.h"

MED_IMAGING_BEGIN_NAMESPACE

class VOIModel;
class VolumeInfos;
class QtWidgets_Export VOIStatisticObserver : public IObserver
{
public:
    VOIStatisticObserver();
    virtual ~VOIStatisticObserver();

    void set_model(std::shared_ptr<VOIModel> model);
    void set_volume_infos(std::shared_ptr<VolumeInfos> volume_infos);

    virtual void update();

protected:

private:
    std::weak_ptr<VOIModel> _model;
    std::shared_ptr<VolumeInfos> _volume_infos;
};

MED_IMAGING_END_NAMESPACE


#endif