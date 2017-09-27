#ifndef MED_IMG_REVIEW_MI_OB_ANNOTATION_STATISTIC_H
#define MED_IMG_REVIEW_MI_OB_ANNOTATION_STATISTIC_H

#include <memory>
#include "mi_review_common.h"
#include "util/mi_observer_interface.h"

MED_IMG_BEGIN_NAMESPACE

class ModelAnnotation;
class VolumeInfos;
class OBAnnotationStatistic : public IObserver
{
public:
    OBAnnotationStatistic();
    virtual ~OBAnnotationStatistic();

    void set_model(std::shared_ptr<ModelAnnotation> model);
    void set_volume_infos(std::shared_ptr<VolumeInfos> volume_infos);

    virtual void update(int code_id = 0);

protected:

private:
    std::weak_ptr<ModelAnnotation> _model;
    std::shared_ptr<VolumeInfos> _volume_infos;
};

MED_IMG_END_NAMESPACE


#endif