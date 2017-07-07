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


using namespace medical_imaging;

namespace
{
    std::shared_ptr<ImageDataHeader> _data_header;
    std::shared_ptr<ImageData> _volume_data;
    std::shared_ptr<VolumeInfos> _volumeinfos;

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
        _scene->place_mpr(SAGITTAL);
        _scene->initialize();

    }

    void Display()
    {
        try
        {
            glViewport(0,0,_width , _height);
            glClearColor(1.0,0.0,0.0,1.0);
            glClear(GL_COLOR_BUFFER_BIT);

            _scene->render(0);

            //glDrawPixels(_width , _height , GL_RGBA , GL_UNSIGNED_BYTE , (void*)_canvas->get_color_array());

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