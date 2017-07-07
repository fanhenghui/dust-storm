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

#include "GL/freeglut.h"


using namespace medical_imaging;

namespace
{
    std::shared_ptr<ImageDataHeader> _data_header;
    std::shared_ptr<ImageData> _volume_data;
    std::shared_ptr<OrthoCamera> _camera;
    std::shared_ptr<OrthoCameraInteractor> m_pCameraInteractor;
    std::shared_ptr<CameraCalculator> m_pCameraCal;
    std::shared_ptr<MPREntryExitPoints> m_pMPREE;
    std::shared_ptr<RayCaster> _ray_caster;
    std::shared_ptr<RayCasterCanvas> _canvas;
    std::shared_ptr<VolumeInfos> m_pVolumeInfos;

    std::shared_ptr<GLTexture1D> m_pPseudoColor;

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

        m_pVolumeInfos.reset( new VolumeInfos());
        m_pVolumeInfos->set_data_header(_data_header);
        m_pVolumeInfos->set_volume(_volume_data);

        _camera.reset(new OrthoCamera());
        m_pCameraCal.reset(new CameraCalculator(_volume_data));
        m_pCameraCal->init_mpr_placement(_camera , TRANSVERSE , Point3(0,0,0));

        m_pCameraInteractor.reset(new OrthoCameraInteractor(_camera));

        m_pMPREE.reset(new MPREntryExitPoints());
        m_pMPREE->set_display_size(_width,_height);
        m_pMPREE->set_camera(_camera);
        m_pMPREE->set_camera_calculator(m_pCameraCal);
        m_pMPREE->set_strategy(GPU_BASE);
        m_pMPREE->set_volume_data(_volume_data);
        m_pMPREE->set_thickness(1.0f);

        _canvas.reset(new RayCasterCanvas());
        _canvas->set_display_size(_width , _height);
        _canvas->initialize();

        _ray_caster.reset(new RayCaster());
        _ray_caster->set_entry_exit_points(m_pMPREE);
        _ray_caster->set_canvas(_canvas);
        _ray_caster->set_camera(_camera);
        _ray_caster->set_volume_data(_volume_data);
        _ray_caster->set_volume_to_world_matrix(m_pCameraCal->get_volume_to_world_matrix());
        _ray_caster->set_sample_rate(1.0);
        _ray_caster->set_global_window_level(252,40+1024);
        _ray_caster->set_strategy(GPU_BASE);
        _ray_caster->set_composite_mode(COMPOSITE_AVERAGE);
        _ray_caster->set_color_inverse_mode(COLOR_INVERSE_DISABLE);

        //GPU
        _ray_caster->set_volume_data_texture(m_pVolumeInfos->get_volume_texture());

        UIDType uid;
        m_pPseudoColor = GLResourceManagerContainer::instance()->get_texture_1d_manager()->create_object(uid);
        m_pPseudoColor->set_description("Pseudo color texture gray");
        m_pPseudoColor->initialize();
        m_pPseudoColor->bind();
        GLTextureUtils::set_1d_wrap_s(GL_CLAMP_TO_EDGE);
        GLTextureUtils::set_filter(GL_TEXTURE_1D , GL_LINEAR);
        unsigned char pData[] = {0,0,0,0,255,255,255,255};
        m_pPseudoColor->load(GL_RGBA8 , 2, GL_RGBA , GL_UNSIGNED_BYTE , pData);

        _ray_caster->set_pseudo_color_texture(m_pPseudoColor , 2);

    }

    void RayCasterCanvasToScreen()
    {
        glBindFramebuffer(GL_READ_FRAMEBUFFER , _canvas->get_fbo()->get_id());
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER , 0);
        glDrawBuffer(GL_BACK);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glBlitFramebuffer(0,0,_width, _height , 0,0,_width , _height , GL_COLOR_BUFFER_BIT , GL_LINEAR);


        /* glEnable(GL_TEXTURE_2D);
        _canvas->get_color_attach_texture()->bind();

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
        glDisable(GL_TEXTURE_2D);

        _canvas->debug_output_color("D:/mpr_rgba.raw");*/
    }

    void Display()
    {
        try
        {
            glViewport(0,0,_width , _height);
            glClearColor(1.0,0.0,0.0,1.0);
            glClear(GL_COLOR_BUFFER_BIT);

            m_pMPREE->calculate_entry_exit_points();
            _ray_caster->render(m_iTestCode);


            RayCasterCanvasToScreen();

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
                std::cout << "W H :" << _width << " " << _height << std::endl;
                m_pMPREE->debug_output_entry_points("D:/entry_exit.rgb.raw");
                break;
            }
        case 'a':
            {
                m_pCameraCal->init_mpr_placement(_camera , TRANSVERSE , Point3(0,0,0));
                m_pCameraInteractor->set_initial_status(_camera);
                m_pCameraInteractor->resize(_width , _height);
                break;
            }
        case 's':
            {
                m_pCameraCal->init_mpr_placement(_camera , SAGITTAL , Point3(0,0,0));
                m_pCameraInteractor->set_initial_status(_camera);
                m_pCameraInteractor->resize(_width , _height);
                break;
            }
        case 'c':
            {
                m_pCameraCal->init_mpr_placement(_camera , CORONAL, Point3(0,0,0));
                m_pCameraInteractor->set_initial_status(_camera);
                m_pCameraInteractor->resize(_width , _height);
                break;
            }
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
        m_pMPREE->set_display_size(_width , _height);
        _canvas->set_display_size(_width , _height);
        _canvas->update_fbo();
        m_pCameraInteractor->resize(_width , _height);
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
            m_pCameraInteractor->rotate(m_ptPre , cur_pt , _width , _height);
            
        }
        else if (m_iButton == GLUT_MIDDLE_BUTTON)
        {
            m_pCameraInteractor->pan(m_ptPre , cur_pt , _width , _height);
        }
        else if (m_iButton == GLUT_RIGHT_BUTTON)
        {
            m_pCameraInteractor->zoom(m_ptPre , cur_pt , _width , _height);
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