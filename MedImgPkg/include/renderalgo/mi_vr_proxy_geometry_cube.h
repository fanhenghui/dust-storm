#ifndef MEDIMGRENDERALGO_VR_PROXY_GEOMETRY_CUBE_H
#define MEDIMGRENDERALGO_VR_PROXY_GEOMETRY_CUBE_H

#include "arithmetic/mi_aabb.h"
#include "glresource/mi_gl_resource_define.h"
#include "glresource/mi_gl_resource_manager_container.h"
#include "renderalgo/mi_render_algo_export.h"
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class VREntryExitPoints;
class ProxyGeometryCube {
public:
    ProxyGeometryCube();
    ~ProxyGeometryCube();

    void initialize();

    void set_vr_entry_exit_poitns(
        std::shared_ptr<VREntryExitPoints> vr_entry_exit_points);

    void calculate_entry_exit_points();

protected:
private:
    GLVAOPtr _gl_vao;
    GLBufferPtr _gl_color_buffer;
    GLBufferPtr _gl_vertex_buffer;
    GLProgramPtr _gl_program;
    GLResourceShield _res_shield;

    std::weak_ptr<VREntryExitPoints> _vr_entry_exit_points;
    AABB _last_aabb;

private:
    DISALLOW_COPY_AND_ASSIGN(ProxyGeometryCube);
};

MED_IMG_END_NAMESPACE

#endif