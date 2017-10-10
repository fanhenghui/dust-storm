#include "mi_vr_proxy_geometry_brick.h"
#include "GL/glew.h"

#include "arithmetic/mi_camera_base.h"

#include "glresource/mi_gl_buffer.h"
#include "glresource/mi_gl_fbo.h"
#include "glresource/mi_gl_program.h"
#include "glresource/mi_gl_resource_manager_container.h"
#include "glresource/mi_gl_utils.h"
#include "glresource/mi_gl_vao.h"

#include "io/mi_image_data.h"

#include "mi_brick_pool.h"
#include "mi_camera_calculator.h"
#include "mi_shader_collection.h"
#include "mi_vr_entry_exit_points.h"
#include "mi_render_algo_logger.h"

MED_IMG_BEGIN_NAMESPACE

ProxyGeometryBrick::ProxyGeometryBrick()
    : _draw_element_count(0), _last_brick_filter_items(-1) {
    _last_aabb._min = Point3::S_ZERO_POINT;
    _last_aabb._max = Point3::S_ZERO_POINT;
}

ProxyGeometryBrick::~ProxyGeometryBrick() {}

void ProxyGeometryBrick::initialize() {
    if (nullptr == _gl_program) {
        UIDType uid;
        _gl_program = GLResourceManagerContainer::instance()
                      ->get_program_manager()
                      ->create_object(uid);
        _gl_program->set_description("proxy geometry brick program");
        _gl_program->initialize();

        _gl_vao = GLResourceManagerContainer::instance()
                  ->get_vao_manager()
                  ->create_object(uid);
        _gl_vao->set_description("proxy geometry brick VAO");
        _gl_vao->initialize();

        _gl_vertex_buffer = GLResourceManagerContainer::instance()
                            ->get_buffer_manager()
                            ->create_object(uid);
        _gl_vertex_buffer->set_description("proxy geometry brick vertex buffer");
        _gl_vertex_buffer->initialize();

        _gl_color_buffer = GLResourceManagerContainer::instance()
                           ->get_buffer_manager()
                           ->create_object(uid);
        _gl_color_buffer->set_description("proxy geometry brick vertex buffer");
        _gl_color_buffer->initialize();

        _gl_element_buffer = GLResourceManagerContainer::instance()
                             ->get_buffer_manager()
                             ->create_object(uid);
        _gl_element_buffer->set_description("proxy geometry brick element buffer");
        _gl_element_buffer->initialize();

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

        _gl_element_buffer->set_buffer_target(GL_ELEMENT_ARRAY_BUFFER);
        _gl_element_buffer->bind();
        _gl_element_buffer->load(0, nullptr, GL_DYNAMIC_DRAW);

        _gl_vao->unbind();

        _res_shield.add_shield<GLVAO>(_gl_vao);
        _res_shield.add_shield<GLBuffer>(_gl_color_buffer);
        _res_shield.add_shield<GLBuffer>(_gl_vertex_buffer);
        _res_shield.add_shield<GLProgram>(_gl_program);
    }
}

void ProxyGeometryBrick::set_vr_entry_exit_poitns(
    std::shared_ptr<VREntryExitPoints> vr_entry_exit_points) {
    _vr_entry_exit_points = vr_entry_exit_points;
}

void ProxyGeometryBrick::calculate_entry_exit_points() {
    std::shared_ptr<VREntryExitPoints> entry_exit_points =
        _vr_entry_exit_points.lock();
    RENDERALGO_CHECK_NULL_EXCEPTION(entry_exit_points);

    if (need_brick_filtering_i()) {
        if (entry_exit_points->_brick_filter_items & BF_MASK) {
            brick_flitering_mask_i();
        } else {
            brick_filtering_non_mask_i();
        }
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

    glDrawElements(GL_TRIANGLES, _draw_element_count, GL_UNSIGNED_INT, NULL);

    CHECK_GL_ERROR;

    // 2 render exit points
    glDrawBuffer(GL_COLOR_ATTACHMENT1);

    glClearDepth(0.0);
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glDepthMask(GL_TRUE);
    glDepthFunc(GL_GEQUAL);

    glDrawElements(GL_TRIANGLES, _draw_element_count, GL_UNSIGNED_INT, NULL);

    _gl_program->unbind();
    _gl_vao->unbind();

    CHECK_GL_ERROR;

    glPopAttrib();

    entry_exit_points->_gl_fbo->unbind();

    CHECK_GL_ERROR;
}

bool ProxyGeometryBrick::need_brick_filtering_i() {
    std::shared_ptr<VREntryExitPoints> entry_exit_points =
        _vr_entry_exit_points.lock();
    RENDERALGO_CHECK_NULL_EXCEPTION(entry_exit_points);

    bool no_changed = true;

    if (_last_aabb != entry_exit_points->_aabb) {
        update_vertex_color_i();
        _last_aabb = entry_exit_points->_aabb;
        no_changed &= false;
    }

    if (_last_window_levels != entry_exit_points->_window_levels) {
        _last_window_levels = entry_exit_points->_window_levels;
        no_changed &= false;
    }

    if (_last_vis_labels != entry_exit_points->_vis_labels) {
        _last_vis_labels = entry_exit_points->_vis_labels;
        no_changed &= false;
    }

    if (_last_brick_filter_items != entry_exit_points->_brick_filter_items) {
        _last_brick_filter_items = entry_exit_points->_brick_filter_items;
        no_changed &= false;
    }

    return !no_changed;
}

void ProxyGeometryBrick::update_vertex_color_i() {
    std::shared_ptr<VREntryExitPoints> entry_exit_points =
        _vr_entry_exit_points.lock();
    RENDERALGO_CHECK_NULL_EXCEPTION(entry_exit_points);

    const std::shared_ptr<BrickPool>& brick_pool = entry_exit_points->_brick_pool;
    const BrickGeometry& brick_geometry = brick_pool->get_brick_geometry();

    std::shared_ptr<ImageData> volume = entry_exit_points->_volume_data;
    RENDERALGO_CHECK_NULL_EXCEPTION(volume);

    if (nullptr == _cur_vertex_array || nullptr == _cur_color_array) {
        _cur_vertex_array.reset(new float[brick_geometry.vertex_count * 3]);
        _cur_color_array.reset(new float[brick_geometry.vertex_count * 4]);
    }

    AABB aabb_wv;
    aabb_wv._min = Point3::S_ZERO_POINT;
    aabb_wv._max = Point3(volume->_dim[0], volume->_dim[1], volume->_dim[2]);

    if (entry_exit_points->_aabb == aabb_wv) {
        memcpy(_cur_vertex_array.get(), brick_geometry.vertex_array,
               brick_geometry.vertex_count * 3 * sizeof(float));
        memcpy(_cur_color_array.get(), brick_geometry.color_array,
               brick_geometry.vertex_count * 4 * sizeof(float));
    } else {
        brick_pool->get_clipping_brick_geometry(entry_exit_points->_aabb,
                                                _cur_vertex_array.get(),
                                                _cur_color_array.get());
    }

    _gl_vertex_buffer->bind();
    _gl_vertex_buffer->load(brick_geometry.vertex_count * 3 * sizeof(float),
                            _cur_vertex_array.get(), GL_STATIC_DRAW);
    _gl_color_buffer->bind();
    _gl_color_buffer->load(brick_geometry.vertex_count * 4 * sizeof(float),
                           _cur_color_array.get(), GL_STATIC_DRAW);
}

void ProxyGeometryBrick::brick_filtering_non_mask_i() {
    MI_RENDERALGO_LOG(MI_TRACE) << "IN proxy geometry brick filtering.";
    std::shared_ptr<VREntryExitPoints> entry_exit_points =
        _vr_entry_exit_points.lock();
    RENDERALGO_CHECK_NULL_EXCEPTION(entry_exit_points);

    std::shared_ptr<ImageData> volume = entry_exit_points->_volume_data;
    RENDERALGO_CHECK_NULL_EXCEPTION(volume);

    const std::shared_ptr<BrickPool>& brick_pool = entry_exit_points->_brick_pool;
    const BrickGeometry& brick_geometry = brick_pool->get_brick_geometry();
    VolumeBrickInfo* volume_brick_info = brick_pool->get_volume_brick_info();
    RENDERALGO_CHECK_NULL_EXCEPTION(volume_brick_info);

    unsigned int brick_dim[3] = {0, 0, 0};
    brick_pool->get_brick_dim(brick_dim);

    std::unique_ptr<unsigned int[]> u_ele_idx_array(
        new unsigned int[brick_pool->get_brick_count() * 36]);
    unsigned int* ele_idx_array = u_ele_idx_array.get();

    auto it_wl = _last_window_levels.find(0);

    if (it_wl == _last_window_levels.end()) {
        RENDERALGO_THROW_EXCEPTION("window level of non-mask is empty");
    }

    float ww = it_wl->second[0] / volume->_slope;
    float wl = (it_wl->second[1] - volume->_intercept) / volume->_slope;
    const float filter_min = wl - ww * 0.5f;

    const int brick_count_layer_jump = brick_dim[0] * brick_dim[1];
    int z_jump(0), zy_jump(0);
    int brick_idx(0);

    AABBI brick_range;
    brick_pool->calculate_intercect_brick_range(_last_aabb, brick_range);

    int reset_brick_count = 0;

    for (int z = brick_range._min[2]; z <= brick_range._max[2]; ++z) {
        z_jump = z * brick_count_layer_jump;

        for (int y = brick_range._min[1]; y <= brick_range._max[1]; ++y) {
            zy_jump = z_jump + y * brick_dim[0];

            for (int x = brick_range._min[0]; x <= brick_range._max[0]; ++x) {
                brick_idx = zy_jump + x;

                if (volume_brick_info[brick_idx].max < filter_min) {
                    continue;
                }

                memcpy(ele_idx_array + reset_brick_count * 36,
                       brick_geometry.brick_idx_units[brick_idx].idx,
                       sizeof(unsigned int) * 36);
                ++reset_brick_count;
            }
        }
    }

    _draw_element_count = reset_brick_count * 36;

    _gl_element_buffer->bind();
    _gl_element_buffer->load(sizeof(unsigned int) * _draw_element_count,
                             ele_idx_array, GL_DYNAMIC_DRAW);
    MI_RENDERALGO_LOG(MI_TRACE) << "OUT proxy geometry brick filtering.";
}

void ProxyGeometryBrick::brick_flitering_mask_i() {
    MI_RENDERALGO_LOG(MI_TRACE) << "IN proxy geometry brick filtering with mask.";

    std::shared_ptr<VREntryExitPoints> entry_exit_points =
        _vr_entry_exit_points.lock();
    RENDERALGO_CHECK_NULL_EXCEPTION(entry_exit_points);

    std::shared_ptr<ImageData> volume = entry_exit_points->_volume_data;
    RENDERALGO_CHECK_NULL_EXCEPTION(volume);

    const std::shared_ptr<BrickPool>& brick_pool = entry_exit_points->_brick_pool;
    const BrickGeometry& brick_geometry = brick_pool->get_brick_geometry();
    VolumeBrickInfo* volume_brick_info = brick_pool->get_volume_brick_info();
    RENDERALGO_CHECK_NULL_EXCEPTION(volume_brick_info);
    MaskBrickInfo* mask_brick_info = brick_pool->get_mask_brick_info(_last_vis_labels);
    RENDERALGO_CHECK_NULL_EXCEPTION(mask_brick_info);

    unsigned int brick_dim[3] = {0, 0, 0};
    brick_pool->get_brick_dim(brick_dim);

    std::unique_ptr<unsigned int[]> u_ele_idx_array(
        new unsigned int[brick_pool->get_brick_count() * 36]);
    unsigned int* ele_idx_array = u_ele_idx_array.get();

    float filter_min[256];
    memset(filter_min , 0 , sizeof(float)*256);
    for (auto it = _last_window_levels.begin(); it != _last_window_levels.end(); ++it)
    {
        unsigned int label = it->first;
        float ww = it->second.get_x() / volume->_slope;
        float wl =  (it->second.get_y() - volume->_intercept) / volume->_slope;
        filter_min[label] = wl - ww*0.5f;
    }

    const int brick_count_layer_jump = brick_dim[0] * brick_dim[1];
    int z_jump(0), zy_jump(0);
    int brick_idx(0);

    AABBI brick_range;
    brick_pool->calculate_intercect_brick_range(_last_aabb, brick_range);

    int reset_brick_count = 0;
    unsigned char brick_label = 0;
    const unsigned char EMPTY_SPACE = 0;
    const unsigned char CHASO_SPACE = 255;
    for (int z = brick_range._min[2]; z <= brick_range._max[2]; ++z) {
        z_jump = z * brick_count_layer_jump;

        for (int y = brick_range._min[1]; y <= brick_range._max[1]; ++y) {
            zy_jump = z_jump + y * brick_dim[0];

            for (int x = brick_range._min[0]; x <= brick_range._max[0]; ++x) {
                brick_idx = zy_jump + x;

                brick_label = static_cast<unsigned char>(mask_brick_info[brick_idx].label);
                if (brick_label == EMPTY_SPACE) {
                    //empty brick
                    continue;
                } else if(brick_label != CHASO_SPACE) {
                    //homogenous brick
                    const float brick_max = volume_brick_info[brick_idx].max;
                    if (brick_max < filter_min[brick_label]) {
                        continue;
                    }
                } else if (brick_label == CHASO_SPACE) {
                    //chaso brick
                }

                memcpy(ele_idx_array + reset_brick_count * 36,
                    brick_geometry.brick_idx_units[brick_idx].idx,
                    sizeof(unsigned int) * 36);
                ++reset_brick_count;
            }
        }
    }

    _draw_element_count = reset_brick_count * 36;

    _gl_element_buffer->bind();
    _gl_element_buffer->load(sizeof(unsigned int) * _draw_element_count,
        ele_idx_array, GL_DYNAMIC_DRAW);
    MI_RENDERALGO_LOG(MI_TRACE) << "OUT proxy geometry brick filtering with mask.";
}

MED_IMG_END_NAMESPACE
