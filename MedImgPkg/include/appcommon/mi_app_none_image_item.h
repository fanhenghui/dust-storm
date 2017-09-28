#ifndef MED_IMG_APPCOMMON_MI_APP_NONE_IMAGE_ITEM_H
#define MED_IMG_APPCOMMON_MI_APP_NONE_IMAGE_ITEM_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "appcommon/mi_app_common_export.h"
#include "arithmetic/mi_volume_statistician.h"
#include "arithmetic/mi_ortho_camera.h"
#include "io/mi_voi.h"


MED_IMG_BEGIN_NAMESPACE

enum NoneImageType {
    InvalidType = -1,
    CornerInfos = 0,
    WindowLevel = 1,
    MPRPage = 2,
    Direction = 3,
    Annotation = 4,
};
class MsgNoneImgCollection;
class SceneBase;
class INoneImg {
public:
    INoneImg( NoneImageType type):_type(type) {};
    virtual ~INoneImg() {};

    virtual void fill_msg(MsgNoneImgCollection* msg) const = 0;
    virtual bool check_dirty() = 0;
    virtual void update() = 0;//should be just update dirty to acc

    NoneImageType get_type() const { return _type;};
    void set_scene(std::shared_ptr<SceneBase> scene) {_scene = scene;};

private:
    NoneImageType _type;

protected:
    std::shared_ptr<SceneBase> _scene;
};

class ModelAnnotation;
class NoneImgAnnotations : public INoneImg {
public:
    struct AnnotationUnit {
        int type;
        int id;
        int status;
        int visibility;
        float para0;//center x
        float para1;//center y
        float para2;//radius
    };

public:
    NoneImgAnnotations():INoneImg(Annotation) ,_pre_width(0), _pre_height(0) {};
    virtual ~NoneImgAnnotations() {};
    virtual void fill_msg(MsgNoneImgCollection* msg) const;
    virtual bool check_dirty();
    virtual void update();

    void add_annotation(const AnnotationUnit& anno);
    void set_annotations(const std::vector<AnnotationUnit> circles);
    const std::vector<AnnotationUnit>& get_annotations() const;

    void set_model(std::shared_ptr<ModelAnnotation> model) {_model = model;};

private:
    std::vector<AnnotationUnit> _annotations;
    std::weak_ptr<ModelAnnotation> _model;

    //cache
    std::vector<VOISphere> _pre_vois;
    //std::vector<IntensityInfo> _pre_intensity_infos;
    OrthoCamera _pre_camera;
    int _pre_width;
    int _pre_height;
};

// LT|1:patientName|2:patientID\n
//just sending once
class NoneImgCornerInfos : public INoneImg {
public:
    enum PositionType {
        LT = 0,
        LB = 1,
        RT = 2,
        RB = 3,
    };

public:
    NoneImgCornerInfos():INoneImg(CornerInfos),_init(false) {};
    virtual ~NoneImgCornerInfos() {};
    virtual void fill_msg(MsgNoneImgCollection* msg) const;
    virtual bool check_dirty();
    virtual void update();

    void set_infos(PositionType pos, std::vector<std::pair<int, std::string>> infos);
    void add_info(PositionType pos, std::pair<int, std::string> info);
    std::string to_string() const;

private:
    std::map<PositionType,std::vector<std::pair<int, std::string>>> _infos;
    bool _init;
};

MED_IMG_END_NAMESPACE


#endif