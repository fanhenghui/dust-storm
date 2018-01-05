#include "mi_vr_entry_exit_points.h"

#include "glresource/mi_gl_fbo.h"
#include "glresource/mi_gl_texture_2d.h"
#include "glresource/mi_gl_texture_cache.h"
#include "glresource/mi_gl_utils.h"

#include "mi_vr_proxy_geometry_brick.h"
#include "mi_vr_proxy_geometry_cube.h"

MED_IMG_BEGIN_NAMESPACE

VREntryExitPoints::VREntryExitPoints() : EntryExitPoints() {}

VREntryExitPoints::~VREntryExitPoints() {}

void VREntryExitPoints::initialize() {
    EntryExitPoints::initialize();

    if (nullptr == _gl_fbo) {
        _gl_fbo = GLResourceManagerContainer::instance()
            ->get_fbo_manager()->create_object("VR entry exit points FBO");
        _gl_fbo->initialize();

        _gl_depth_texture = GLResourceManagerContainer::instance()
            ->get_texture_2d_manager()->create_object("VR entry exit points FBO depth texture");
        _gl_depth_texture->initialize();

        _gl_fbo->bind();
        _gl_fbo->set_target(GL_FRAMEBUFFER);

        _entry_points_texture->bind();
        GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_BORDER);
        GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
        _entry_points_texture->load(GL_RGBA32F, _width, _height, GL_RGBA, GL_FLOAT,
                                    NULL);
        _gl_fbo->attach_texture(GL_COLOR_ATTACHMENT0, _entry_points_texture);

        _exit_points_texture->bind();
        GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_BORDER);
        GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
        _exit_points_texture->load(GL_RGBA32F, _width, _height, GL_RGBA, GL_FLOAT,
                                   NULL);
        _gl_fbo->attach_texture(GL_COLOR_ATTACHMENT1, _exit_points_texture);

        _gl_depth_texture->bind();
        GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_BORDER);
        GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
        _gl_depth_texture->load(GL_DEPTH_COMPONENT16, _width, _height,
                                GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, NULL);
        _gl_fbo->attach_texture(GL_DEPTH_ATTACHMENT, _gl_depth_texture);

        _gl_fbo->unbind();

        _proxy_geo_cube.reset(new ProxyGeometryCube);
        _proxy_geo_cube->set_vr_entry_exit_poitns(
            std::dynamic_pointer_cast<VREntryExitPoints>(shared_from_this()));
        _proxy_geo_cube->initialize();

        _proxy_geo_brick.reset(new ProxyGeometryBrick);
        _proxy_geo_brick->set_vr_entry_exit_poitns(
            std::dynamic_pointer_cast<VREntryExitPoints>(shared_from_this()));
        _proxy_geo_brick->initialize();

        _res_shield.add_shield<GLFBO>(_gl_fbo);
        _res_shield.add_shield<GLTexture2D>(_gl_depth_texture);
    }
}

void VREntryExitPoints::set_display_size(int width, int height) {
    EntryExitPoints::set_display_size(width, height);

    if (GPU_BASE == _strategy && _gl_depth_texture) {
        // _gl_depth_texture->bind();
        // GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_BORDER);
        // GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
        // _gl_depth_texture->load(GL_DEPTH_COMPONENT16, _width, _height,
        //                         GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, NULL);
        // _gl_depth_texture->unbind();

        GLTextureCache::instance()->cache_load(
            GL_TEXTURE_2D, _gl_depth_texture, GL_CLAMP_TO_BORDER, GL_LINEAR,
            GL_DEPTH_COMPONENT16, _width, _height, 0, GL_DEPTH_COMPONENT,
            GL_UNSIGNED_SHORT, nullptr);
    }
}

void VREntryExitPoints::set_brick_pool(std::shared_ptr<BrickPool> brick_pool) {
    _brick_pool = brick_pool;
}

void VREntryExitPoints::set_bounding_box(const AABB& aabb) {
    _aabb = aabb;
}

void VREntryExitPoints::set_clipping_plane(std::vector<Plane> planes) {
    // TODO vr cull logic
}

void VREntryExitPoints::set_visible_labels(const std::vector<unsigned char>& labels) {
    _vis_labels = labels;
}

void VREntryExitPoints::set_window_level(float ww, float wl,
        unsigned char label,
        bool global /*= false*/) {
    if (global) {
        _window_levels.clear();
        _window_levels[0] = Vector2f(ww, wl);
    } else {
        if (_window_levels.find(0) !=
                _window_levels.end()) { // clear global window level (label 0)
            _window_levels.clear();
        }
        _window_levels[label] = Vector2f(ww, wl);
    }
}

void VREntryExitPoints::set_brick_cull_item(int items) {
    _brick_cull_items = items;
}

void VREntryExitPoints::set_brick_filter_item(int items) {
    _brick_filter_items = items;
}

void VREntryExitPoints::set_proxy_geometry(ProxyGeometry pg) {
    _proxy_geometry = pg;
}

void VREntryExitPoints::calculate_entry_exit_points() {
    initialize();

    if (_proxy_geometry == PG_CUBE) {
        _proxy_geo_cube->calculate_entry_exit_points();
    } else if (_proxy_geometry == PG_BRICKS) {
        _proxy_geo_brick->calculate_entry_exit_points();
    }
}

MED_IMG_END_NAMESPACE
