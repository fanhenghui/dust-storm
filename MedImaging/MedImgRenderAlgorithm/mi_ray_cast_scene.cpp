#include "mi_ray_cast_scene.h"


#include "MedImgCommon/mi_configuration.h"

#include "MedImgArithmetic/mi_ortho_camera.h"
#include "MedImgArithmetic/mi_point2.h"

#include "MedImgIO/mi_image_data.h"
#include "MedImgIO/mi_image_data_header.h"

#include "MedImgGLResource/mi_gl_texture_2d.h"
#include "MedImgGLResource/mi_gl_fbo.h"
#include "MedImgGLResource/mi_gl_utils.h"
#include "MedImgGLResource/mi_gl_texture_1d.h"

#include "MedImgRenderAlgorithm/mi_camera_calculator.h"
#include "MedImgRenderAlgorithm/mi_camera_interactor.h"
#include "MedImgRenderAlgorithm/mi_volume_infos.h"
#include "MedImgRenderAlgorithm/mi_entry_exit_points.h"
#include "MedImgRenderAlgorithm/mi_ray_caster.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_canvas.h"


MED_IMAGING_BEGIN_NAMESPACE

RayCastScene::RayCastScene():SceneBase(),_test_code(0),_global_ww(0),_global_wl(0)
{
    _ray_cast_camera.reset(new OrthoCamera());
    _camera = _ray_cast_camera;

    _camera_interactor.reset(new OrthoCameraInteractor(_ray_cast_camera));

    _ray_caster.reset(new RayCaster());

    _canvas.reset(new RayCasterCanvas());


    if (CPU == Configuration::instance()->get_processing_unit_type())
    {
        _ray_caster->set_strategy(CPU_BASE);
    }
    else
    {
        _ray_caster->set_strategy(GPU_BASE);
    }
}

RayCastScene::RayCastScene(int width , int height):SceneBase(width , height),_test_code(0),
    _global_ww(0),
    _global_wl(0)
{
    _ray_cast_camera.reset(new OrthoCamera());
    _camera = _ray_cast_camera;

    _camera_interactor.reset(new OrthoCameraInteractor(_ray_cast_camera));

    _ray_caster.reset(new RayCaster());

    _canvas.reset(new RayCasterCanvas());
    _canvas->set_display_size(_width , _height);


    if (CPU == Configuration::instance()->get_processing_unit_type())
    {
        _ray_caster->set_strategy(CPU_BASE);
    }
    else
    {
        _ray_caster->set_strategy(GPU_BASE);
    }
}

RayCastScene::~RayCastScene()
{

}

void RayCastScene::initialize()
{
    SceneBase::initialize();

    //Canvas
    _canvas->initialize();
}

void RayCastScene::finalize()
{
    _canvas->finialize();
    _entry_exit_points->finialize();
    _ray_caster->finialize();
}

void RayCastScene::set_display_size(int width , int height)
{
    SceneBase::set_display_size(width ,height);
    _canvas->set_display_size(width , height);
    _canvas->update_fbo();//update texture size
    _entry_exit_points->set_display_size(width , height);
    _camera_interactor->resize(width , height);
}

void RayCastScene::render(int test_code)
{
    //Skip render scene
    if (!get_dirty())
    {
        return;
    }

    //////////////////////////////////////////////////////////////////////////
    //TODO other common graphic object rendering list

    //////////////////////////////////////////////////////////////////////////
    //1 Ray casting
    CHECK_GL_ERROR;
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glViewport(0,0,_width , _height);
    glClearColor(0,0,0,0);
    glClear(GL_COLOR_BUFFER_BIT);

    _entry_exit_points->calculate_entry_exit_points();
    _ray_caster->render(_test_code);

    glPopAttrib();
    CHECK_GL_ERROR;

    //////////////////////////////////////////////////////////////////////////
    //2 Mapping ray casting result to Scene FBO
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glViewport(0,0,_width , _height);

    _scene_fbo->bind();
    glDrawBuffer(GL_COLOR_ATTACHMENT0);

    CHECK_GL_ERROR;

    glEnable(GL_TEXTURE_2D);
    _canvas->get_color_attach_texture()->bind();

    CHECK_GL_ERROR;

    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); 
    glVertex2f(-1.0, -1.0);
    glTexCoord2f(1.0, 0.0); 
    glVertex2f(1.0, -1.0);
    glTexCoord2f(1.0, 1.0); 
    glVertex2f(1.0, 1.0);
    glTexCoord2f(0.0, 1.0);
    glVertex2f(-1.0, 1.0);
    glEnd();

    //glDisable(GL_TEXTURE_2D);
    CHECK_GL_ERROR;
    glPopAttrib();//TODO Here will give a GL_INVALID_OPERATION error !!!
    CHECK_GL_ERROR;

    set_dirty(false);
}

void RayCastScene::set_volume_infos(std::shared_ptr<VolumeInfos> volume_infos)
{
    try
    {
        RENDERALGO_CHECK_NULL_EXCEPTION(volume_infos);
        _volume_infos = volume_infos;

        std::shared_ptr<ImageData> pVolume = _volume_infos->get_volume();
        RENDERALGO_CHECK_NULL_EXCEPTION(pVolume);

        std::shared_ptr<ImageDataHeader> data_header = _volume_infos->get_data_header();
        RENDERALGO_CHECK_NULL_EXCEPTION(data_header);

        //Camera calculator
        _camera_calculator = volume_infos->get_camera_calculator();

        //Entry exit points
        _entry_exit_points->set_image_data(pVolume);
        _entry_exit_points->set_camera(_camera);
        _entry_exit_points->set_display_size(_width , _height);
        _entry_exit_points->set_camera_calculator(_camera_calculator);
        _entry_exit_points->initialize();

        //Ray caster
        _ray_caster->set_canvas(_canvas);
        _ray_caster->set_entry_exit_points(_entry_exit_points);
        _ray_caster->set_camera(_camera);
        _ray_caster->set_volume_to_world_matrix(_camera_calculator->get_volume_to_world_matrix());
        _ray_caster->set_volume_data(pVolume);
        //_ray_caster->set_mask_data(pMask);

        if (GPU == Configuration::instance()->get_processing_unit_type())
        {
            //initialize gray pseudo color texture
            if (!_pseudo_color_texture)
            {
                UIDType uid;
                _pseudo_color_texture = GLResourceManagerContainer::instance()->get_texture_1d_manager()->create_object(uid);
                _pseudo_color_texture->set_description("Pseudo color texture gray");
                _pseudo_color_texture->initialize();
                _pseudo_color_texture->bind();
                GLTextureUtils::set_1d_wrap_s(GL_CLAMP_TO_EDGE);
                GLTextureUtils::set_filter(GL_TEXTURE_1D , GL_LINEAR);
                unsigned char pData[] = {0,0,0,0,255,255,255,255};
                _pseudo_color_texture->load(GL_RGBA8 , 2, GL_RGBA , GL_UNSIGNED_BYTE , pData);
            }
            _ray_caster->set_pseudo_color_texture(_pseudo_color_texture , 2);

            //Volume texture
            _ray_caster->set_volume_data_texture(volume_infos->get_volume_texture());
        }
        _ray_caster->initialize();

        set_dirty(true);
    }
    catch (const Exception& e)
    {
        //TOOD LOG
        std::cout << e.what();
        //assert(false);
        throw e;
    }
}

void RayCastScene::set_mask_label_level(LabelLevel label_level)
{
    _ray_caster->set_mask_label_level(label_level);
    set_dirty(true);
}

void RayCastScene::set_sample_rate(float sample_rate)
{
    _ray_caster->set_sample_rate(sample_rate);
    set_dirty(true);
}

void RayCastScene::set_visible_labels(std::vector<unsigned char> labels)
{
    _ray_caster->set_visible_labels(labels);
    set_dirty(true);
}

void RayCastScene::set_window_level(float ww , float wl , unsigned char label)
{
    RENDERALGO_CHECK_NULL_EXCEPTION(_volume_infos);
    _volume_infos->get_volume()->regulate_wl(ww , wl);

    _ray_caster->set_window_level(ww , wl , label);

    set_dirty(true);
}

void RayCastScene::set_global_window_level(float ww , float wl)
{
    _global_ww = ww;
    _global_wl = wl;

    RENDERALGO_CHECK_NULL_EXCEPTION(_volume_infos);
    _volume_infos->get_volume()->regulate_wl(ww , wl);

    _ray_caster->set_global_window_level(ww, wl);

    set_dirty(true);
}

void RayCastScene::set_mask_mode(MaskMode mode)
{
    _ray_caster->set_mask_mode(mode);

    set_dirty(true);
}

void RayCastScene::set_composite_mode(CompositeMode mode)
{
    _ray_caster->set_composite_mode(mode);

    set_dirty(true);
}

void RayCastScene::set_interpolation_mode(InterpolationMode mode)
{
    _ray_caster->set_interpolation_mode(mode);

    set_dirty(true);
}

void RayCastScene::set_shading_mode(ShadingMode mode)
{
    _ray_caster->set_shading_mode(mode);

    set_dirty(true);
}

void RayCastScene::set_color_inverse_mode(ColorInverseMode mode)
{
    _ray_caster->set_color_inverse_mode(mode);

    set_dirty(true);
}

void RayCastScene::set_test_code(int test_code)
{
    _test_code = test_code;

    set_dirty(true);
}

void RayCastScene::get_global_window_level(float& ww , float& wl) const
{
    ww = _global_ww;
    wl = _global_wl;
}

std::shared_ptr<VolumeInfos> RayCastScene::get_volume_infos() const
{
    return _volume_infos;
}

std::shared_ptr<CameraCalculator> RayCastScene::get_camera_calculator() const
{
    return _camera_calculator;
}



MED_IMAGING_END_NAMESPACE