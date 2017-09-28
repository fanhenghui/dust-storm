#ifndef MED_IMG_APPCOMMON_MI_APP_NONE_IMAGE_H
#define MED_IMG_APPCOMMON_MI_APP_NONE_IMAGE_H

#include <memory>
#include <map>
#include <set>
#include "appcommon/mi_app_none_image_interface.h"
#include "appcommon/mi_app_none_image_item.h"

MED_IMG_BEGIN_NAMESPACE

class INoneImg;
class AppNoneImage : public IAppNoneImage {
public:
    AppNoneImage();
    virtual ~AppNoneImage();

    virtual bool check_dirty();
    virtual void update();
    virtual char* serialize_dirty(int& buffer_size) const;

    void add_none_image_item(std::shared_ptr<INoneImg> none_image);
    std::shared_ptr<INoneImg> get_none_image_item(NoneImageType type);

private:
    std::map<NoneImageType, std::shared_ptr<INoneImg>> _none_image_items;
    std::set<NoneImageType> _dirty_cache;
};

MED_IMG_END_NAMESPACE
#endif