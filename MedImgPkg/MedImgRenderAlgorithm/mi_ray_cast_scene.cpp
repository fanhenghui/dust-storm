#include "mi_ray_cast_scene.h"


#include "MedImgUtil/mi_configuration.h"
#include "MedImgUtil/mi_file_util.h"

#include "MedImgArithmetic/mi_ortho_camera.h"
#include "MedImgArithmetic/mi_point2.h"

#include "MedImgIO/mi_image_data.h"
#include "MedImgIO/mi_image_data_header.h"

#include "MedImgGLResource/mi_gl_texture_2d.h"
#include "MedImgGLResource/mi_gl_fbo.h"
#include "MedImgGLResource/mi_gl_utils.h"
#include "MedImgGLResource/mi_gl_texture_1d.h"
#include "MedImgGLResource/mi_gl_resource_manager_container.h"
#include "MedImgGLResource/mi_gl_texture_cache.h"

#include "MedImgRenderAlgorithm/mi_camera_calculator.h"
#include "MedImgRenderAlgorithm/mi_camera_interactor.h"
#include "MedImgRenderAlgorithm/mi_volume_infos.h"
#include "MedImgRenderAlgorithm/mi_entry_exit_points.h"
#include "MedImgRenderAlgorithm/mi_ray_caster.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_canvas.h"


MED_IMG_BEGIN_NAMESPACE

RayCastScene::RayCastScene():SceneBase(),_global_ww(0),_global_wl(0)
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

RayCastScene::RayCastScene(int width , int height):SceneBase(width , height),
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
    _entry_exit_points->initialize();
    _ray_caster->initialize();
}

void RayCastScene::set_display_size(int width , int height)
{
    SceneBase::set_display_size(width ,height);
    _canvas->set_display_size(width , height);
    _canvas->update_fbo();//update texture size
    _entry_exit_points->set_display_size(width , height);
    _camera_interactor->resize(width , height);
}

void RayCastScene::pre_render()
{
    //refresh volume & mask & their infos
    _volume_infos->refresh();

    //scene FBO , ray casting program ...
    initialize();

    //entry exit points initialize
    _entry_exit_points->initialize();

    //GL resource update (discard)
    GLResourceManagerContainer::instance()->update_all();

    //GL texture udpate
    GLTextureCache::instance()->process_cache();
}

void RayCastScene::render()
{
    pre_render();

    //Skip render scene
    if (!get_dirty())
    {
        return;
    }

    CHECK_GL_ERROR;

    //////////////////////////////////////////////////////////////////////////
    //TODO other common graphic object rendering list

    //////////////////////////////////////////////////////////////////////////
    //1 Ray casting
    //glPushAttrib(GL_ALL_ATTRIB_BITS);

    glViewport(0,0,_width , _height);
    glClearColor(0,0,0,0);
    glClear(GL_COLOR_BUFFER_BIT);

    _entry_exit_points->calculate_entry_exit_points();

    _ray_caster->render();
    //glPopAttrib();

    //////////////////////////////////////////////////////////////////////////
    //2 Mapping ray casting result to Scene FBO (<><>flip vertically<><>)
    //glPushAttrib(GL_ALL_ATTRIB_BITS);

    glViewport(0,0,_width , _height);

    _scene_fbo->bind();
    glDrawBuffer(GL_COLOR_ATTACHMENT0);

    glEnable(GL_TEXTURE_2D);
    _canvas->get_color_attach_texture()->bind();

    glBegin(GL_QUADS);
    // glTexCoord2f(0.0, 0.0); 
    // glVertex2f(-1.0, -1.0);
    // glTexCoord2f(1.0, 0.0); 
    // glVertex2f(1.0, -1.0);
    // glTexCoord2f(1.0, 1.0); 
    // glVertex2f(1.0, 1.0);
    // glTexCoord2f(0.0, 1.0);
    // glVertex2f(-1.0, 1.0);

    glTexCoord2f(0.0, 1.0); 
    glVertex2f(-1.0, -1.0);
    glTexCoord2f(1.0, 1.0); 
    glVertex2f(1.0, -1.0);
    glTexCoord2f(1.0, 0.0); 
    glVertex2f(1.0, 1.0);
    glTexCoord2f(0.0, 0.0);
    glVertex2f(-1.0, 1.0);
    
    glEnd();

    //CHECK_GL_ERROR;
    //glPopAttrib();//TODO Here will give a GL_INVALID_OPERATION error !!!
    //CHECK_GL_ERROR;

    _scene_fbo->unbind();

    CHECK_GL_ERROR;
    

//CHECK_GL_ERROR;
    //_canvas->debug_output_color("/home/wr/data/output.raw");
//CHECK_GL_ERROR;


    // _scene_color_attach_0->bind();
    // std::unique_ptr<unsigned char[]> color_array(new unsigned char[_width*_height*3]);
    // _scene_color_attach_0->download(GL_RGB , GL_UNSIGNED_BYTE , color_array.get());

    // FileUtil::write_raw("/home/wr/data/scene_output_rgb.raw" , (char*)color_array.get(), _width*_height*4);
    

    set_dirty(false);
}

void RayCastScene::set_volume_infos(std::shared_ptr<VolumeInfos> volume_infos)
{
    try
    {
        RENDERALGO_CHECK_NULL_EXCEPTION(volume_infos);
        _volume_infos = volume_infos;

        std::shared_ptr<ImageData> volume = _volume_infos->get_volume();
        RENDERALGO_CHECK_NULL_EXCEPTION(volume);

        std::shared_ptr<ImageData> mask  = _volume_infos->get_mask();
        RENDERALGO_CHECK_NULL_EXCEPTION(mask);

        std::shared_ptr<ImageDataHeader> data_header = _volume_infos->get_data_header();
        RENDERALGO_CHECK_NULL_EXCEPTION(data_header);

        //Camera calculator
        _camera_calculator = volume_infos->get_camera_calculator();

        //Entry exit points
        _entry_exit_points->set_volume_data(volume);
        _entry_exit_points->set_camera(_camera);
        _entry_exit_points->set_display_size(_width , _height);
        _entry_exit_points->set_camera_calculator(_camera_calculator);

        //Ray caster
        _ray_caster->set_canvas(_canvas);
        _ray_caster->set_entry_exit_points(_entry_exit_points);
        _ray_caster->set_camera(_camera);
        _ray_caster->set_volume_to_world_matrix(_camera_calculator->get_volume_to_world_matrix());
        _ray_caster->set_volume_data(volume);
        _ray_caster->set_mask_data(mask);

        if (GPU == Configuration::instance()->get_processing_unit_type())
        {
            //initialize gray pseudo color texture
            if (!_pseudo_color_texture)
            {
                UIDType uid;
                _pseudo_color_texture = GLResourceManagerContainer::instance()->get_texture_1d_manager()->create_object(uid);
                _pseudo_color_texture->set_description("pseudo color texture gray");
                _res_shield.add_shield<GLTexture1D>(_pseudo_color_texture);

                unsigned char* gray_array = new unsigned char[8];
                gray_array[0] = 0;
                gray_array[1] = 0;
                gray_array[2] = 0;
                gray_array[3] = 0;
                gray_array[4] = 255;
                gray_array[5] = 255;
                gray_array[6] = 255;
                gray_array[7] = 255;
                GLTextureCache::instance()->cache_load(GL_TEXTURE_1D , _pseudo_color_texture , 
                    GL_CLAMP_TO_EDGE ,GL_LINEAR , GL_RGBA8 , 2, 0,0, GL_RGBA , GL_UNSIGNED_BYTE , (char*)gray_array);
            }

            //set texture
            _ray_caster->set_pseudo_color_texture(_pseudo_color_texture , 2);
            _ray_caster->set_volume_data_texture(volume_infos->get_volume_texture());
            _ray_caster->set_mask_data_texture(volume_infos->get_mask_texture());
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
    if(_ray_caster->get_visible_labels() != labels)
    {
        _ray_caster->set_visible_labels(labels);
        set_dirty(true); 
    }
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
    if (_ray_caster->get_mask_mode() != mode)
    {
        _ray_caster->set_mask_mode(mode);
        set_dirty(true);
    }
}

void RayCastScene::set_composite_mode(CompositeMode mode)
{
    if (_ray_caster->get_composite_mode() != mode)
    {
        _ray_caster->set_composite_mode(mode);
        set_dirty(true);
    }
}

void RayCastScene::set_interpolation_mode(InterpolationMode mode)
{
    if (_ray_caster->get_interpolation_mode() != mode)
    {
        _ray_caster->set_interpolation_mode(mode);
        set_dirty(true);
    }
}

void RayCastScene::set_shading_mode(ShadingMode mode)
{
    if (_ray_caster->get_shading_mode() != mode)
    {
        _ray_caster->set_shading_mode(mode);
        set_dirty(true);
    }
}

void RayCastScene::set_color_inverse_mode(ColorInverseMode mode)
{
    if (_ray_caster->get_color_inverse_mode() != mode)
    {
        _ray_caster->set_color_inverse_mode(mode);
        set_dirty(true);
    }
}

void RayCastScene::set_test_code(int test_code)
{
    _ray_caster->set_test_code(test_code);

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

MED_IMG_END_NAMESPACE