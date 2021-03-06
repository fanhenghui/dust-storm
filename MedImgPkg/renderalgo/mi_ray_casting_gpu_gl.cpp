#include "mi_ray_casting_gpu_gl.h"

#include "glresource/mi_gl_buffer.h"
#include "glresource/mi_gl_program.h"
#include "glresource/mi_gl_utils.h"
#include "glresource/mi_gl_vao.h"
#include "glresource/mi_gl_time_query.h"

#include "mi_ray_caster.h"
#include "mi_rc_step_base.h"
#include "mi_rc_step_color_inverse.h"
#include "mi_rc_step_composite.h"
#include "mi_rc_step_main.h"
#include "mi_rc_step_mask_overlay.h"
#include "mi_rc_step_mask_sampler.h"
#include "mi_rc_step_ray_casting.h"
#include "mi_rc_step_shading.h"
#include "mi_rc_step_utils.h"
#include "mi_rc_step_volume_sampler.h"

MED_IMG_BEGIN_NAMESPACE

RayCastingGPUGL::RayCastingGPUGL(std::shared_ptr<RayCaster> ray_caster)
    : _ray_caster(ray_caster), _active_texture_counter(new GLActiveTextureCounter()), 
    _render_duration(0.0f), _last_test_code(-1) {}

RayCastingGPUGL::~RayCastingGPUGL() {}

void RayCastingGPUGL::render() {
    update_i();

    CHECK_GL_ERROR;

    ScopedGLTimeQuery inner_timer(_time_query, &_render_duration);

    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glDisable(GL_DEPTH_TEST);
    glDepthMask(false);

    _program->bind();
    _vao->bind();

    _active_texture_counter->reset();

    for (auto it = _ray_casting_steps.begin(); it != _ray_casting_steps.end(); ++it) {
        (*it)->set_gpu_parameter();
    }

    glDrawArrays(GL_TRIANGLES, 0, 6);

    _vao->unbind();
    _program->unbind();
    glPopAttrib();

    CHECK_GL_ERROR;
}

void RayCastingGPUGL::update_i() {
    CHECK_GL_ERROR;

    // Create VAO
    if (!_vao) {
        _vao = GLResourceManagerContainer::instance()
            ->get_vao_manager()->create_object("GPU ray casting VAO");
        _vao->initialize();

        _buffer_vertex = GLResourceManagerContainer::instance()
            ->get_buffer_manager()->create_object("GPU ray casting vertex buffer (-1 -1)~ (1,1)");
        _buffer_vertex->initialize();
        _buffer_vertex->set_buffer_target(GL_ARRAY_BUFFER);

        _vao->bind();
        _buffer_vertex->bind();
        const float vertex[] = { -1, 1,  0, -1, -1, 0, 1,  -1, 0,
                           1,  -1, 0, 1,  1,  0, -1, 1,  0};
        _buffer_vertex->load(sizeof(vertex), vertex, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(0);

        _vao->unbind();
        _buffer_vertex->unbind();

        //create time query
        _time_query = GLResourceManagerContainer::instance()->get_time_query_manager()->create_object("GPU ray casting time query");
        _time_query->initialize();

        _res_shield.add_shield<GLVAO>(_vao);
        _res_shield.add_shield<GLBuffer>(_buffer_vertex);
        _res_shield.add_shield<GLTimeQuery>(_time_query);
    }

    CHECK_GL_ERROR;

    // Create Program
    if (!_program) {
        _program = GLResourceManagerContainer::instance()
            ->get_program_manager()->create_object("GPU ray casting program");
        _program->initialize();
        _res_shield.add_shield<GLProgram>(_program);
    }

    std::shared_ptr<RayCaster> ray_caster = _ray_caster.lock();

    const int test_code = ray_caster->get_test_code();

    if (_ray_casting_steps.empty() || _mask_mode != ray_caster->_mask_mode ||
            _composite_mode != ray_caster->_composite_mode ||
            _interpolation_mode != ray_caster->_interpolation_mode ||
            _shading_mode != ray_caster->_shading_mode ||
            _color_inverse_mode != ray_caster->_color_inverse_mode ||
            _mask_overlay_mode != ray_caster->_mask_overlay_mode ||
            _last_test_code != test_code) {
        _ray_casting_steps.clear();
        _mask_mode = ray_caster->_mask_mode;
        _composite_mode = ray_caster->_composite_mode;
        _interpolation_mode = ray_caster->_interpolation_mode;
        _shading_mode = ray_caster->_shading_mode;
        _color_inverse_mode = ray_caster->_color_inverse_mode;
        _mask_overlay_mode = ray_caster->_mask_overlay_mode;
        _last_test_code = test_code;

#define STEP_PUSH_BACK(step_class_name)                                        \
  _ray_casting_steps.push_back(std::shared_ptr<step_class_name>(               \
      new step_class_name(ray_caster, _program)));

        if (_last_test_code == 1 || _last_test_code == 2) {
            // Main
            STEP_PUSH_BACK(RCStepMainVert);
            STEP_PUSH_BACK(RCStepMainTestFrag);
        } else {
            // Main
            STEP_PUSH_BACK(RCStepMainVert);
            STEP_PUSH_BACK(RCStepMainFrag);

            // Utils
            STEP_PUSH_BACK(RCStepUtils);

            // Composite
            if (_composite_mode == COMPOSITE_DVR) {
                STEP_PUSH_BACK(RCStepRayCastingDVR);
                STEP_PUSH_BACK(RCStepCompositeDVR);
            } else if (_composite_mode == COMPOSITE_AVERAGE) {
                STEP_PUSH_BACK(RCStepRayCastingAverage);
                STEP_PUSH_BACK(RCStepCompositeAverage);
            } else if (_composite_mode == COMPOSITE_MIP) {
                STEP_PUSH_BACK(RCStepRayCastingMIPMinIP);
                STEP_PUSH_BACK(RCStepCompositeMIP);
            } else if (_composite_mode == COMPOSITE_MINIP) {
                STEP_PUSH_BACK(RCStepRayCastingMIPMinIP);
                STEP_PUSH_BACK(RCStepCompositeMinIP);
            }

            // Mask
            if (_mask_mode == MASK_NONE) {
                STEP_PUSH_BACK(RCStepMaskNoneSampler);
            } else if (_mask_mode == MASK_MULTI_LABEL) {
                STEP_PUSH_BACK(RCStepMaskNearstSampler);
            } else if (_mask_mode == MASK_MULTI_LINEAR_LABEL) {
                STEP_PUSH_BACK(RCStepMaskLinearSampler);
            }

            // Volume
            if (_interpolation_mode == LINEAR) {
                STEP_PUSH_BACK(RCStepVolumeLinearSampler);
            } else if (_interpolation_mode == NEARST) {
                STEP_PUSH_BACK(RCStepVolumeNearstSampler);
            } else if (_interpolation_mode == CUBIC) {
                // TODO
            }

            // Shading
            if (_shading_mode == SHADING_NONE) {
                STEP_PUSH_BACK(RCStepShadingNone);
            } else if (_shading_mode == SHADING_PHONG) {
                STEP_PUSH_BACK(RCStepShadingPhong);
            }

            // Color inverse
            if (_color_inverse_mode == COLOR_INVERSE_DISABLE) {
                STEP_PUSH_BACK(RCStepColorInverseDisable);
            } else { //(_color_inverse_mode == COLOR_INVERSE_ENABLE)
                STEP_PUSH_BACK(RCStepColorInverseEnable);
            }

            // Overlay mask label
            if (_mask_overlay_mode == MASK_OVERLAY_DISABLE) {
                STEP_PUSH_BACK(RCStepMaskOverlayDisable);
            } else {
                STEP_PUSH_BACK(RCStepMaskOverlayEnable);
            }
        }

        // compile
        std::vector<GLShaderInfo> shaders;

        for (auto it = _ray_casting_steps.begin(); it != _ray_casting_steps.end(); ++it) {
            shaders.push_back((*it)->get_shader_info());
        }

        _program->finalize();
        _program->initialize();
        _program->set_shaders(shaders);
        _program->compile();

        for (auto it = _ray_casting_steps.begin(); it != _ray_casting_steps.end(); ++it) {
            (*it)->set_active_texture_counter(_active_texture_counter);
            (*it)->get_uniform_location();
        }
    }

#undef STEP_PUSH_BACK

    CHECK_GL_ERROR;
}

float RayCastingGPUGL::get_rendering_duration() const {
    return _render_duration;
}

MED_IMG_END_NAMESPACE