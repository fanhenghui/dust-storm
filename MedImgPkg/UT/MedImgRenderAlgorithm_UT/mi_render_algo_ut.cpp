#include "GL/glew.h"

#include "MedImgUtil/mi_configuration.h"
#include "MedImgUtil/mi_file_util.h"

#include "MedImgIO/mi_dicom_loader.h"
#include "MedImgIO/mi_image_data.h"
#include "MedImgIO/mi_image_data_header.h"

#include "MedImgArithmetic/mi_ortho_camera.h"

#include "MedImgGLResource/mi_gl_fbo.h"
#include "MedImgGLResource/mi_gl_texture_2d.h"
#include "MedImgGLResource/mi_gl_environment.h"
#include "MedImgGLResource/mi_gl_texture_1d.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "MedImgRenderAlgorithm/mi_camera_calculator.h"
#include "MedImgRenderAlgorithm/mi_mpr_entry_exit_points.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_canvas.h"
#include "MedImgRenderAlgorithm/mi_ray_caster.h"
#include "MedImgRenderAlgorithm/mi_camera_interactor.h"
#include "MedImgRenderAlgorithm/mi_volume_infos.h"
#include "MedImgRenderAlgorithm/mi_mpr_scene.h"

#include "GL/freeglut.h"
#include "libgpujpeg/gpujpeg.h"
#include "libgpujpeg/gpujpeg_common.h"
#include "cuda_runtime.h"


using namespace medical_imaging;

#define cuda_check_error(msg) \
    { \
        cudaError_t err = cudaGetLastError(); \
        if( cudaSuccess != err) { \
            fprintf(stderr, "[GPUJPEG] [Error] %s (line %i): %s: %s.\n", \
                __FILE__, __LINE__, msg, cudaGetErrorString( err) ); \
        } \
    } \

namespace
{
    std::shared_ptr<ImageDataHeader> _data_header;
    std::shared_ptr<ImageData> _volume_data;
    std::shared_ptr<VolumeInfos> _volumeinfos;
    gpujpeg_opengl_texture* gpujpeg_texture;

    gpujpeg_parameters param;//gpujpeg parameter 
    gpujpeg_image_parameters param_image;//image parameter
    gpujpeg_encoder* encoder = nullptr;//jpeg encoder

    std::shared_ptr<MPRScene> _scene;

    int _width = 800;
    int _height = 800;
    int m_iButton = -1;
    Point2 m_ptPre;
    int m_iTestCode = 0;

    std::vector<std::string> GetFiles()
    {

        const std::string root  = "/home/wr/data/AB_CTA_01/";
        std::vector<std::string> files;
        FileUtil::get_all_file_recursion(root , std::vector<std::string>() , files);
        return files;
    }

    void Init()
    {
        Configuration::instance()->set_processing_unit_type(GPU);
        GLUtils::set_check_gl_flag(true);

        std::vector<std::string> files = GetFiles();
        DICOMLoader loader;
        IOStatus status = loader.load_series(files , _volume_data , _data_header);

        _volumeinfos.reset( new VolumeInfos());
        _volumeinfos->set_data_header(_data_header);
        _volumeinfos->set_volume(_volume_data);

        //Create empty mask
        std::shared_ptr<ImageData> mask_data(new ImageData());
        _volume_data->shallow_copy(mask_data.get());
        mask_data->_channel_num = 1;
        mask_data->_data_type = medical_imaging::UCHAR;
        mask_data->mem_allocate();
        _volumeinfos->set_mask(mask_data);


        _scene.reset(new MPRScene(_width , _height));
        const float PRESET_CT_LUNGS_WW = 1500;
        const float PRESET_CT_LUNGS_WL = -400;

        _scene->set_volume_infos(_volumeinfos);
        _scene->set_sample_rate(1.0);
        _scene->set_global_window_level(PRESET_CT_LUNGS_WW,PRESET_CT_LUNGS_WL);
        _scene->set_composite_mode(COMPOSITE_AVERAGE);
        _scene->set_color_inverse_mode(COLOR_INVERSE_DISABLE);
        _scene->set_mask_mode(MASK_NONE);
        _scene->set_interpolation_mode(LINEAR);
        _scene->place_mpr(TRANSVERSE);
        _scene->initialize();

        ////////////////////////////////////////
        //init gpujpeg
        gpujpeg_init_device(0,0);
        unsigned int tex_id = _scene->get_scene_color_attach_0()->get_id();
        std::cout << "scene base texture id : " << tex_id << std::endl;
        gpujpeg_texture = gpujpeg_opengl_texture_register(tex_id, GPUJPEG_OPENGL_TEXTURE_READ);
        std::cout << "Cuda graphics resource : " << gpujpeg_texture->texture_pbo_resource << std::endl;

        //init gpujepg parameter
        gpujpeg_set_default_parameters(&param);//默认参数
        gpujpeg_parameters_chroma_subsampling(&param);//默认采样参数;
        
        //也可以自己设置采样参数
        // param.sampling_factor[0].horizontal = 4;
        // param.sampling_factor[0].vertical = 4;
        // param.sampling_factor[1].horizontal = 1;
        // param.sampling_factor[1].vertical = 2;
        // param.sampling_factor[2].horizontal = 2;
        // param.sampling_factor[2].vertical = 1;

        //Init gpujpeg image parameter
        gpujpeg_image_set_default_parameters(&param_image);
        param_image.width = _width;
        param_image.height = _height;
        param_image.comp_count = 3;
        param_image.color_space = GPUJPEG_RGB;
        param_image.sampling_factor = GPUJPEG_4_4_4;

        //create encoder
         encoder = gpujpeg_encoder_create(&param,&param_image);
         if (!encoder){
             std::cout << "Create encoder failed!\n";
         }
         cuda_check_error("error");
        
    }

    void Display()
    {
        try
        {
            glViewport(0,0,_width , _height);
            glClearColor(0.0,0.0,0.0,1.0);
            glClear(GL_COLOR_BUFFER_BIT);
            
            //_scene->initialize();
            _scene->render(0);
            _scene->render_to_back();
            // glBindFramebuffer(GL_DRAW_FRAMEBUFFER , 0);
            // _scene->download_image_buffer();
            // _scene->swap_image_buffer();
            // unsigned char* buffer = nullptr;
            // _scene->get_image_buffer(buffer);
            // FileUtil::write_raw("/home/wr/data/output_ut.raw",buffer , _width*_height*4);

            //glDrawPixels(_width , _height , GL_RGBA , GL_UNSIGNED_BYTE , (void*)_canvas->get_color_array());


            //Encoding input
            gpujpeg_encoder_input encoder_input;
            
            ////////////////////////////////////////////////////////
            //方案1 download下来再压缩
            //Test download scene FBO color attachment 0
            //GLTexture2DPtr scene_color_attach_0 = _scene->get_scene_color_attach_0();
            //scene_color_attach_0->bind();
            //std::unique_ptr<unsigned char[]> color_array(new unsigned char[_width*_height*3]);
            //scene_color_attach_0->download(GL_RGB , GL_UNSIGNED_BYTE , color_array.get());
            //FileUtil::write_raw("/home/wr/data/scene_output_rgb.raw" , (char*)color_array.get(), _width*_height*3);

            cuda_check_error("error");

            //gpujpeg_encoder_input_set_image(&encoder_input, color_array.get());


            ////////////////////////////////////////////////////////
            //方案2 直接用texture来做
            gpujpeg_encoder_input_set_texture(&encoder_input, gpujpeg_texture);

            cuda_check_error("error");

            uint8_t* image_compressed = nullptr;
            int image_compressed_size = 0;
            if (gpujpeg_encoder_encode(encoder, &encoder_input, &image_compressed,
                &image_compressed_size) != 0){
                std::cout << "encode failed!\n";
            }

            cuda_check_error("error");

            std::cout << "compress image size : " << image_compressed_size << std::endl;

            if (gpujpeg_image_save_to_file("/home/wr/data/scene_output_rgb.jpeg", image_compressed,
            image_compressed_size) != 0){
                std::cout << "save filed failed!\n";
            }

            cuda_check_error("error");

            //gpujpeg_image_destroy(image_compressed);

            cuda_check_error("error");

            //gpujpeg_encoder_destroy(encoder);
            image_compressed = nullptr;

            glutSwapBuffers();
        }
        catch (Exception& e)
        {
            std::cout << e.what();
            abort();
        }
    }

    void Keyboard(unsigned char key , int x , int y)
    {
        switch(key)
        {
        case 't':
            {
                // std::cout << "W H :" << _width << " " << _height << std::endl;
                // m_pMPREE->debug_output_entry_points("D:/entry_exit.rgb.raw");
                _scene->page(1);
                break;
            }
        // case 'a':
        //     {
        //         m_pCameraCal->init_mpr_placement(_camera , TRANSVERSE , Point3(0,0,0));
        //         m_pCameraInteractor->set_initial_status(_camera);
        //         m_pCameraInteractor->resize(_width , _height);
        //         break;
        //     }
        // case 's':
        //     {
        //         m_pCameraCal->init_mpr_placement(_camera , SAGITTAL , Point3(0,0,0));
        //         m_pCameraInteractor->set_initial_status(_camera);
        //         m_pCameraInteractor->resize(_width , _height);
        //         break;
        //     }
        // case 'c':
        //     {
        //         m_pCameraCal->init_mpr_placement(_camera , CORONAL, Point3(0,0,0));
        //         m_pCameraInteractor->set_initial_status(_camera);
        //         m_pCameraInteractor->resize(_width , _height);
        //         break;
        //     }
        case 'f':
            {
                m_iTestCode = 1- m_iTestCode;
                break;
            }
        default:
            break;
        }

        glutPostRedisplay();
    }

    void resize(int x , int y)
    {
        _width = x;
        _height = y;
        _scene->set_display_size(_width , _height);
        if(encoder){
            gpujpeg_encoder_destroy(encoder);
            encoder = gpujpeg_encoder_create(&param,&param_image);
            if (!encoder){
                std::cout << "Create encoder failed!\n";
            }
        }
        glutPostRedisplay();
    }

    void Idle()
    {
        glutPostRedisplay();
    }

    void MouseClick(int button , int status , int x , int y)
    {
        x = x< 0 ? 0 : x;
        x = x> _width-1 ?  _width-1 : x;
        y = y< 0 ? 0 : y;
        y = y> _height-1 ?  _height-1 : y;

        m_iButton = button;

        if (m_iButton == GLUT_LEFT_BUTTON)
        {
            
        }
        else if (m_iButton == GLUT_MIDDLE_BUTTON)
        {

        }
        else if (m_iButton == GLUT_RIGHT_BUTTON)
        {

        }

        
        m_ptPre = Point2(x,y);
        glutPostRedisplay();
    }

    void MouseMotion(int x , int y)
    {
        x = x< 0 ? 0 : x;
        x = x> _width-1 ?  _width-1 : x;
        y = y< 0 ? 0 : y;
        y = y> _height-1 ?  _height-1 : y;

        Point2 cur_pt(x,y);

        //std::cout << "Pre : " << m_ptPre.x << " " <<m_ptPre.y << std::endl;
        //std::cout << "Cur : " << cur_pt.x << " " <<cur_pt.y << std::endl;
        if (m_iButton == GLUT_LEFT_BUTTON)
        {
            _scene->rotate(m_ptPre , cur_pt);
            
        }
        else if (m_iButton == GLUT_MIDDLE_BUTTON)
        {
            _scene->pan(m_ptPre , cur_pt);
        }
        else if (m_iButton == GLUT_RIGHT_BUTTON)
        {
            _scene->zoom(m_ptPre , cur_pt);
        }

        m_ptPre = cur_pt;
        glutPostRedisplay();

    }
}

int main(int argc , char* argv[])
{
    try
    {
        glutInit(&argc , argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
        glutInitWindowPosition(0,0);
        glutInitWindowSize(_width,_height);

        glutCreateWindow("Test GL resource");

        if ( GLEW_OK != glewInit())
        {
            std::cout <<"Init glew failed!\n";
            return -1;
        }

        GLEnvironment env;
        int major , minor;
        env.get_gl_version(major , minor);

        glutDisplayFunc(Display);
        glutReshapeFunc(resize);
        glutIdleFunc(Idle);
        glutKeyboardFunc(Keyboard);
        glutMouseFunc(MouseClick);
        glutMotionFunc(MouseMotion); 

        Init();

        glutMainLoop(); 
        return 0;
    }
    catch (const Exception& e)
    {
        std::cout  << e.what();
        return -1;
    }
}