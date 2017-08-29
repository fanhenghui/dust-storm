#ifndef MED_IMG_OBSERVER_VOI_STATISTIC_H
#define MED_IMG_OBSERVER_VOI_STATISTIC_H

#include "qtpackage/mi_qt_package_export.h"
#include "util/mi_observer_interface.h"

MED_IMG_BEGIN_NAMESPACE

class VOIModel;
class VolumeInfos;
class QtPackage_Export VOIStatisticObserver : public IObserver {
public:
    VOIStatisticObserver();
    virtual ~VOIStatisticObserver();

    void set_model(std::shared_ptr<VOIModel> model);
    void set_volume_infos(std::shared_ptr<VolumeInfos> volume_infos);

    virtual void update(int code_id = 0);

protected:

private:
    std::weak_ptr<VOIModel> _model;
    std::shared_ptr<VolumeInfos> _volume_infos;
};

MED_IMG_END_NAMESPACE


#endif