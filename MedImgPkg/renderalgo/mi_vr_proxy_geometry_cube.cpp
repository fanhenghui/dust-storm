#include "mi_vr_proxy_geometry_cube.h"
#include "GL/glew.h"

#include "arithmetic/mi_camera_base.h"

#include "glresource/mi_gl_buffer.h"
#include "glresource/mi_gl_fbo.h"
#include "glresource/mi_gl_program.h"
#include "glresource/mi_gl_resource_manager_container.h"
#include "glresource/mi_gl_utils.h"
#include "glresource/mi_gl_vao.h"

#include "mi_camera_calculator.h"
#include "mi_shader_collection.h"
#include "mi_vr_entry_exit_points.h"

MED_IMG_BEGIN_NAMESPACE

ProxyGeometryCube::ProxyGeometryCube() {
    _last_aabb._min = Point3::S_ZERO_POINT;
    _last_aabb._max = Point3::S_ZERO_POINT;
}

ProxyGeometryCube::~ProxyGeometryCube() {}

void ProxyGeometryCube::initialize() {
    if (nullptr == _gl_program) {
        UIDType uid;
        _gl_program = GLResourceManagerContainer::instance()
                      ->get_program_manager()
                      ->create_object(uid);
        _gl_program->set_description("proxy geometry cube program");
        _gl_program->initialize();

        _gl_vao = GLResourceManagerContainer::instance()
                  ->get_vao_manager()
                  ->create_object(uid);
        _gl_vao->set_description("proxy geometry cube VAO");
        _gl_vao->initialize();

        _gl_vertex_buffer = GLResourceManagerContainer::instance()
                            ->get_buffer_manager()
                            ->create_object(uid);
        _gl_vertex_buffer->set_description("proxy geometry cube vertex buffer");
        _gl_vertex_buffer->initialize();

        _gl_color_buffer = GLResourceManagerContainer::instance()
                           ->get_buffer_manager()
                           ->create_object(uid);
        _gl_color_buffer->set_description("proxy geometry cube vertex buffer");
        _gl_color_buffer->initialize();

        // program
        std::vector<GLShaderInfo> shaders;
        shaders.push_back(GLShaderInfo(GL_VERTEX_SHADER,
                                       S_VR_ENTRY_EXIT_POINTS_VERTEX,
                                       "proxy geometry cube vertex shader"));
        shaders.push_back(GLShaderInfo(GL_FRAGMENT_SHADER,
                                       S_VR_ENTRY_EXIT_POINTS_FRAG,
                                       "proxy geometry cube fragment shader"));
        _gl_program->set_shaders(shaders);
        _gl_program->compile();

        // VAO
        _gl_vao->bind();

        _gl_vertex_buffer->set_buffer_target(GL_ARRAY_BUFFER);
        _gl_vertex_buffer->bind();
        _gl_vertex_buffer->load(0, nullptr, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(0);

        _gl_color_buffer->set_buffer_target(GL_ARRAY_BUFFER);
        _gl_color_buffer->bind();
        _gl_color_buffer->load(0, nullptr, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(1);

        _gl_vao->unbind();

        _res_shield.add_shield<GLVAO>(_gl_vao);
        _res_shield.add_shield<GLBuffer>(_gl_color_buffer);
        _res_shield.add_shield<GLBuffer>(_gl_vertex_buffer);
        _res_shield.add_shield<GLProgram>(_gl_program);
    }
}

void ProxyGeometryCube::set_vr_entry_exit_poitns(
    std::shared_ptr<VREntryExitPoints> vr_entry_exit_points) {
    _vr_entry_exit_points = vr_entry_exit_points;
}

void ProxyGeometryCube::calculate_entry_exit_points() {
    std::shared_ptr<VREntryExitPoints> entry_exit_points =
        _vr_entry_exit_points.lock();
    RENDERALGO_CHECK_NULL_EXCEPTION(entry_exit_points);

    // update VAO
    if (_last_aabb != entry_exit_points->_aabb) {
        _last_aabb = entry_exit_points->_aabb;
        const float pt_min[3] = {(float)_last_aabb._min.x, (float)_last_aabb._min.y,
                                 (float)_last_aabb._min.z
                                };
        const float pt_max[3] = {(float)_last_aabb._max.x, (float)_last_aabb._max.y,
                                 (float)_last_aabb._max.z
                                };

#define VERTEX(pt0, pt1, pt2) pt0[0], pt1[1], pt2[2]

        float vertex[] = {
            VERTEX(pt_max, pt_min, pt_max), VERTEX(pt_max, pt_min, pt_min),
            VERTEX(pt_max, pt_max, pt_min), VERTEX(pt_max, pt_min, pt_max),
            VERTEX(pt_max, pt_max, pt_min), VERTEX(pt_max, pt_max, pt_max),
            VERTEX(pt_max, pt_max, pt_max), VERTEX(pt_max, pt_max, pt_min),
            VERTEX(pt_min, pt_max, pt_min), VERTEX(pt_max, pt_max, pt_max),
            VERTEX(pt_min, pt_max, pt_min), VERTEX(pt_min, pt_max, pt_max),
            VERTEX(pt_max, pt_max, pt_max), VERTEX(pt_min, pt_max, pt_max),
            VERTEX(pt_min, pt_min, pt_max), VERTEX(pt_max, pt_max, pt_max),
            VERTEX(pt_min, pt_min, pt_max), VERTEX(pt_max, pt_min, pt_max),
            VERTEX(pt_min, pt_min, pt_max), VERTEX(pt_min, pt_max, pt_max),
            VERTEX(pt_min, pt_max, pt_min), VERTEX(pt_min, pt_min, pt_max),
            VERTEX(pt_min, pt_max, pt_min), VERTEX(pt_min, pt_min, pt_min),
            VERTEX(pt_min, pt_min, pt_max), VERTEX(pt_min, pt_min, pt_min),
            VERTEX(pt_max, pt_min, pt_min), VERTEX(pt_min, pt_min, pt_max),
            VERTEX(pt_max, pt_min, pt_min), VERTEX(pt_max, pt_min, pt_max),
            VERTEX(pt_min, pt_min, pt_min), VERTEX(pt_min, pt_max, pt_min),
            VERTEX(pt_max, pt_max, pt_min), VERTEX(pt_min, pt_min, pt_min),
            VERTEX(pt_max, pt_max, pt_min), VERTEX(pt_max, pt_min, pt_min)
        };

#undef VERTEX

        float color[36 * 4];

        for (int i = 0; i < 36; ++i) {
            color[i * 4] = vertex[i * 3];
            color[i * 4 + 1] = vertex[i * 3 + 1];
            color[i * 4 + 2] = vertex[i * 3 + 2];
            color[i * 4 + 3] = 0.0f;
        }

        _gl_vertex_buffer->bind();
        _gl_vertex_buffer->load(36 * 3 * sizeof(float), vertex, GL_DYNAMIC_DRAW);

        _gl_color_buffer->bind();
        _gl_color_buffer->load(36 * 4 * sizeof(float), color, GL_DYNAMIC_DRAW);
    }

    CHECK_GL_ERROR;

    entry_exit_points->_gl_fbo->bind();

    glPushAttrib(GL_ALL_ATTRIB_BITS);

    _gl_vao->bind();
    _gl_program->bind();

    CHECK_GL_ERROR;

    const Matrix4 mat_vp =
        entry_exit_points->get_camera()->get_view_projection_matrix();
    const Matrix4 mat_m =
        entry_exit_points->get_camera_calculator()->get_volume_to_world_matrix();
    const Matrix4 mat_mvp = mat_vp * mat_m;

    int loc = _gl_program->get_uniform_location("mat_mvp");

    if (-1 == loc) {
        RENDERALGO_THROW_EXCEPTION("get uniform mat_mvp failed!");
    }

    float mat_mvp_f[16];
    mat_mvp.to_float16(mat_mvp_f);
    glUniformMatrix4fv(loc, 1, GL_FALSE, mat_mvp_f);

    CHECK_GL_ERROR;

    glEnable(GL_DEPTH_TEST);

    // 1 render entry points
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glClearDepth(1.0);
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LEQUAL);

    glDrawArrays(GL_TRIANGLES, 0, 36);

    CHECK_GL_ERROR;

    // 2 render exit points
    glDrawBuffer(GL_COLOR_ATTACHMENT1);

    glClearDepth(0.0);
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glDepthMask(GL_TRUE);
    glDepthFunc(GL_GEQUAL);

    glDrawArrays(GL_TRIANGLES, 0, 36);

    _gl_program->unbind();
    _gl_vao->unbind();

    CHECK_GL_ERROR;

    glPopAttrib();

    entry_exit_points->_gl_fbo->unbind();

    CHECK_GL_ERROR;
}

MED_IMG_END_NAMESPACE
