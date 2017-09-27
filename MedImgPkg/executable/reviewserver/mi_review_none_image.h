#ifndef MED_IMG_MI_REVIEW_NONE_IMAGE_H
#define MED_IMG_MI_REVIEW_NONE_IMAGE_H

#include <memory>
#include "mi_review_common.h"
#include "appcommon/mi_app_none_image_interface.h"

MED_IMG_BEGIN_NAMESPACE

class NoneImgCollection;
class SceneBase;
class VolumeInfos;
class ReviewNoneImage : public IAppNoneImage {
public:
    ReviewNoneImage();
    virtual ~ReviewNoneImage();

    virtual void update();
    virtual char* serialize_dirty(int& buffer_size) const;

    void initialize(std::shared_ptr<VolumeInfos> volume_infos, std::shared_ptr<SceneBase> scene);
    //TODO set model

protected:
    virtual bool check_dirty_i();

private:
    std::shared_ptr<NoneImgCollection> _none_img_collection;
    std::shared_ptr<VolumeInfos> _volume_infos;
    std::shared_ptr<SceneBase> _scene;

    bool _fix_corner_infos_dirty;
    bool _wl_dirty;
    bool _wl_page_dirty;
    //bool _annotation_dirty;
};

MED_IMG_END_NAMESPACE
#endif