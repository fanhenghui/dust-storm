#include "mi_app_none_image.h"
#include "renderalgo/mi_scene_base.h"
#include "renderalgo/mi_volume_infos.h"
#include "renderalgo/mi_vr_scene.h"
#include "renderalgo/mi_mpr_scene.h"
#include "io/mi_image_data_header.h"
#include "mi_app_none_image_item.h"

MED_IMG_BEGIN_NAMESPACE

AppNoneImage::AppNoneImage() {

}

AppNoneImage::~AppNoneImage() {

}

void AppNoneImage::update() {
    //save wl cache to update
}

char* AppNoneImage::serialize_dirty(int& buffer_size) const {
    NoneImgCollection msgcoll;
    if(_fix_corner_infos_dirty) {
        std::shared_ptr<NoneImgCornerInfos> noneimg_cinfos(new NoneImgCornerInfos());
        //TODO set corner based on config file
        APPCOMMON_CHECK_NULL_EXCEPTION(_volume_infos);
        std::shared_ptr<ImageDataHeader> header = _volume_infos->get_data_header();
        
        // patient descriptor
        noneimg_cinfos->add_info(NoneImgCornerInfos::LT, std::make_pair(0, header->patient_name));
        noneimg_cinfos->add_info(NoneImgCornerInfos::LT, std::make_pair(1, header->patient_id));

        // parameters that can be tuned
        noneimg_cinfos->add_info(NoneImgCornerInfos::LB, std::make_pair(0, "Current Slice "));
        noneimg_cinfos->add_info(NoneImgCornerInfos::LB, std::make_pair(1, "W 100 L 100"));
        
        // volume structure descriptor
        std::stringstream ss;
        ss << header->columns << " " << header->rows << " " << header->slice_location.size();
        noneimg_cinfos->add_info(NoneImgCornerInfos::RT, std::make_pair(0, ss.str()));
        
        ss.str(std::string());
        ss << header->slice_thickness;
        noneimg_cinfos->add_info(NoneImgCornerInfos::RT, std::make_pair(1, ss.str()));

        // volume physical descriptor
        ss.str(std::string());
        ss << header->kvp;
        noneimg_cinfos->add_info(NoneImgCornerInfos::RB, std::make_pair(0, ss.str()));

        msgcoll.set_corner_infos(noneimg_cinfos);
    }

    //TODO wl page annotation
    return msgcoll.serialize_to_array(buffer_size);
}

void AppNoneImage::initialize(std::shared_ptr<VolumeInfos> volume_infos, std::shared_ptr<SceneBase> scene) {
    _volume_infos = volume_infos;
    _scene = scene;

    _fix_corner_infos_dirty = true;
    _wl_dirty = true;
    std::shared_ptr<MPRScene> mpr_scene = std::dynamic_pointer_cast<MPRScene>(scene);
    if (mpr_scene) {
        _wl_page_dirty = true;
    }

    this->set_dirty(true);
}

bool AppNoneImage::check_dirty_i() {
    //TODO check wl&paging&annotatio ...
    return false;
}

MED_IMG_END_NAMESPACE