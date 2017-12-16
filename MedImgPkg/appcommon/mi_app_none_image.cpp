#include "mi_app_none_image.h"

#include "io/mi_image_data_header.h"
#include "io/mi_protobuf.h"

#include "renderalgo/mi_scene_base.h"
#include "renderalgo/mi_volume_infos.h"
#include "renderalgo/mi_vr_scene.h"
#include "renderalgo/mi_mpr_scene.h"

#include "mi_app_none_image_item.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

AppNoneImage::AppNoneImage() {

}

AppNoneImage::~AppNoneImage() {

}

void AppNoneImage::update() {
    boost::mutex::scoped_lock locker(_mutex);
    for (auto it = _dirty_cache.begin(); it != _dirty_cache.end(); ++it) {
        _none_image_items[*it]->update();
    }
}

char* AppNoneImage::serialize_dirty(int& buffer_size) const {
    boost::mutex::scoped_lock locker(_mutex);
    MsgNoneImgCollection msg;
    for (auto it = _dirty_cache.begin(); it != _dirty_cache.end(); ++it) {
        typedef std::map<NoneImageType, std::shared_ptr<INoneImg>>::const_iterator const_iter;
        const_iter it_item = _none_image_items.find(*it);
        if (it_item != _none_image_items.end()) {
            it_item->second->fill_msg(&msg);
        }
    }

    char* data = nullptr;
    if (0 != protobuf_serialize(msg,data,buffer_size)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "serialize none-img-collection: serialize failed.";
        return nullptr; 
    } else {
        msg.Clear();
        return data;
    }
}

bool AppNoneImage::check_dirty() {
    boost::mutex::scoped_lock locker(_mutex);
    int dirty_items = 0;
    _dirty_cache.clear();
    for (auto it = _none_image_items.begin(); it != _none_image_items.end(); ++it) {
        if (it->second->check_dirty()) {
            ++dirty_items;    
            _dirty_cache.insert(it->first);
        }        
    }
    return dirty_items !=0 ;
}

void AppNoneImage::add_none_image_item(std::shared_ptr<INoneImg> none_image) {
    boost::mutex::scoped_lock locker(_mutex);
    _none_image_items[none_image->get_type()] = none_image;
}

std::shared_ptr<INoneImg> AppNoneImage::get_none_image_item(NoneImageType type) {
    auto it = _none_image_items.find(type);
    if (it == _none_image_items.end()) {
        return nullptr;
    } else {
        return it->second;
    }
}

MED_IMG_END_NAMESPACE