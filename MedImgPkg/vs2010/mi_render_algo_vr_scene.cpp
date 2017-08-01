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
#include "MedImgGLResource/mi_gl_time_query.h"
#include "MedImgGLResource/mi_gl_resource_manager_container.h"

#include "MedImgRenderAlgorithm/mi_camera_calculator.h"
#include "MedImgRenderAlgorithm/mi_mpr_entry_exit_points.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_canvas.h"
#include "MedImgRenderAlgorithm/mi_ray_caster.h"
#include "MedImgRenderAlgorithm/mi_camera_interactor.h"
#include "MedImgRenderAlgorithm/mi_volume_infos.h"
#include "MedImgRenderAlgorithm/mi_vr_scene.h"

#ifdef WIN32
#include "GL/glut.h"
#else
#include "GL/freeglut.h"
#include "cuda_runtime.h"
#endif

#include "libgpujpeg/gpujpeg.h"
#include "libgpujpeg/gpujpeg_common.h"


using namespace medical_imaging;
namespace
{
    std::shared_ptr<ImageDataHeader> _data_header;
    std::shared_ptr<ImageData> _volume_data;
    std::shared_ptr<VolumeInfos> _volumeinfos;

    std::shared_ptr<VRScene> _scene;

    std::shared_ptr<GLTimeQuery> _time_query;
    std::shared_ptr<GLTimeQuery> _time_query2;

    int _width = 512;
    int _height = 512;
    int _iButton = -1;
    Point2 _ptPre;
    int _iTestCode = 0;

    std::vector<std::string> GetFiles()
    {
#ifdef WIN32
        const std::string root  = "E:/data/MyData/AB_CTA_01/";
#else
        const std::string root  = "/home/wr/data/AB_CTA_01/";
#endif
        
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


        _scene.reset(new VRScene(_width , _height));
        const float PRESET_CT_LUNGS_WW = 1500;
        const float PRESET_CT_LUNGS_WL = -400;

        _scene->set_volume_infos(_volumeinfos);
        _scene->set_sample_rate(1.0);
        _scene->set_global_window_level(PRESET_CT_LUNGS_WW,PRESET_CT_LUNGS_WL);
        _scene->set_composite_mode(COMPOSITE_MIP);
        _scene->set_color_inverse_mode(COLOR_INVERSE_DISABLE);
        _scene->set_mask_mode(MASK_NONE);
        _scene->set_interpolation_mode(LINEAR);
        _scene->set_proxy_geometry(PG_CUBE);
        _scene->set_test_code(_iTestCode);
        _scene->initialize();

        //Time query
        UIDType uid = 0;
        _time_query = GLResourceManagerContainer::instance()->get_time_query_manager()->create_object(uid);
        _time_query->initialize();

        _time_query2 = GLResourceManagerContainer::instance()->get_time_query_manager()->create_object(uid);
        _time_query2->initialize();
        
    }

    void Display()
    {
        try
        {
            glViewport(0,0,_width , _height);
            glClearColor(0.0,0.0,0.0,1.0);
            glClear(GL_COLOR_BUFFER_BIT);
            
            //_time_query->begin();
            _scene->set_dirty(true);

            _scene->render();

            _scene->render_to_back();

            

            //_time_query2->begin();
            //glBindFramebuffer(GL_DRAW_FRAMEBUFFER , 0);
            
           /* _scene->download_image_buffer();
            _scene->swap_image_buffer();
            unsigned char* buffer = nullptr;
            int buffer_size=0;
            _scene->get_image_buffer(buffer,buffer_size);*/

#ifdef WIN32
            //FileUtil::write_raw("D:/temp/output_ut.jpeg",buffer , buffer_size);
#else
            //FileUtil::write_raw("/home/wr/data/output_ut.jpeg",buffer , buffer_size);
#endif
            //std::cout << "compressing time : " << _scene->get_compressing_time() << std::endl;

            //std::cout << "gl compressing time : " << _time_query2->end() << std::endl;

            //std::cout << "rendering time : " << _time_query->end() << std::endl;

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
        case 'f':
            {
                _iTestCode = 1- _iTestCode;
                break;
            }
        case 'r':
            {
                _scene.reset(new VRScene(_width , _height));
                const float PRESET_CT_LUNGS_WW = 1500;
                const float PRESET_CT_LUNGS_WL = -400;

                _scene->set_volume_infos(_volumeinfos);
                _scene->set_sample_rate(1.0);
                _scene->set_global_window_level(PRESET_CT_LUNGS_WW,PRESET_CT_LUNGS_WL);
                _scene->set_composite_mode(COMPOSITE_MIP);
                _scene->set_color_inverse_mode(COLOR_INVERSE_DISABLE);
                _scene->set_mask_mode(MASK_NONE);
                _scene->set_interpolation_mode(LINEAR);
                _scene->set_proxy_geometry(PG_CUBE);
                _scene->set_test_code(_iTestCode);
                _scene->initialize();
                _scene->set_display_size(_width , _height);
                break;
            }
        default:
            break;
        }

        glutPostRedisplay();
    }

    void resize(int x , int y)
    {
        if(x == 0  || y == 0){
            return;
        }

        _width = x;
        _height = y;
        _scene->set_display_size(_width , _height);
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

        _iButton = button;

        if (_iButton == GLUT_LEFT_BUTTON)
        {
            
        }
        else if (_iButton == GLUT_MIDDLE_BUTTON)
        {

        }
        else if (_iButton == GLUT_RIGHT_BUTTON)
        {

        }

        
        _ptPre = Point2(x,y);
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
        if (_iButton == GLUT_LEFT_BUTTON)
        {
            _scene->rotate(_ptPre , cur_pt);
            
        }
        else if (_iButton == GLUT_MIDDLE_BUTTON)
        {
            _scene->pan(_ptPre , cur_pt);
        }
        else if (_iButton == GLUT_RIGHT_BUTTON)
        {
            _scene->zoom(_ptPre , cur_pt);
        }

        _ptPre = cur_pt;
        glutPostRedisplay();

    }
}

int TE_VRScene(int argc , char* argv[])
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