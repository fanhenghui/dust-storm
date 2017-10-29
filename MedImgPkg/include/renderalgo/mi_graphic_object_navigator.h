#ifndef MEDIMGRENDERALGO_MI_GRIPHIC_OBJECT_NAVIGATOR_H
#define MEDIMGRENDERALGO_MI_GRIPHIC_OBJECT_NAVIGATOR_H

#include "renderalgo/mi_graphic_object_interface.h"
#include "glresource/mi_gl_resource_manager_container.h"

MED_IMG_BEGIN_NAMESPACE

class GraphicObjectNavigator : public IGraphicObject {
public:
    GraphicObjectNavigator();
    virtual ~GraphicObjectNavigator();

    virtual void initialize();
    virtual void render(int code = 0);
    void set_scene_display_size(int width, int height);
    void set_navi_position(int x, int y, int width, int height);

private:
    GLTexture2DPtr _tex_map;
    GLResourceShield _res_shield;
    GLVAOPtr _vao;
    int _s_width;
    int _s_height;
    int _x;
    int _y;
    int _width;
    int _height;
};

MED_IMG_END_NAMESPACE
#endif