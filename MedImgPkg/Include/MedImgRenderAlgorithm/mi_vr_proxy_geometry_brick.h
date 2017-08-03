#ifndef MED_IMG_VR_PROXY_GEOMETRY_BRICK_H_
#define MED_IMG_VR_PROXY_GEOMETRY_BRICK_H_

#include <memory>
#include "MedImgRenderAlgorithm/mi_render_algo_export.h"
#include "MedImgArithmetic/mi_aabb.h"
#include "MedImgArithmetic/mi_vector2f.h"
#include "MedImgGLResource/mi_gl_resource_define.h"
#include "MedImgGLResource/mi_gl_resource_manager_container.h"

MED_IMG_BEGIN_NAMESPACE

class VREntryExitPoints;
class ProxyGeometryBrick
{
public:
    ProxyGeometryBrick();
    ~ProxyGeometryBrick();

    void initialize();

    void set_vr_entry_exit_poitns(std::shared_ptr<VREntryExitPoints> vr_entry_exit_points);

    void calculate_entry_exit_points();

private:
    bool need_brick_filtering_i();

    void brick_flitering_mask_i();

    void brick_filtering_non_mask_i();

private:
    GLVAOPtr _gl_vao;
    GLBufferPtr _gl_color_buffer;
    GLBufferPtr _gl_vertex_buffer;
    GLBufferPtr _gl_element_buffer;
    GLProgramPtr _gl_program;
    GLResourceShield _res_shield;

    std::weak_ptr<VREntryExitPoints> _vr_entry_exit_points;

    unsigned int _draw_element_count;

    //cache
    AABB _last_aabb;
    std::map<unsigned char , Vector2f> _last_window_levels;
    std::vector<unsigned char> _last_vis_labels;
    int _last_brick_filter_items;

private:
    DISALLOW_COPY_AND_ASSIGN(ProxyGeometryBrick);
};

MED_IMG_END_NAMESPACE

#endif