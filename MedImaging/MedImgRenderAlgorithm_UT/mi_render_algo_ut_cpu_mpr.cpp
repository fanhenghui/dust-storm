#include "gl/glew.h"

#include "MedImgCommon/mi_concurrency.h"
#include "MedImgCommon/mi_configuration.h"

#include "MedImgIO/mi_dicom_loader.h"
#include "MedImgIO/mi_image_data.h"
#include "MedImgIO/mi_image_data_header.h"

#include "MedImgArithmetic/mi_ortho_camera.h"

#include "MedImgGLResource/mi_gl_fbo.h"
#include "MedImgGLResource/mi_gl_texture_2d.h"
#include "MedImgGLResource/mi_gl_environment.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "MedImgRenderAlgorithm/mi_camera_calculator.h"
#include "MedImgRenderAlgorithm/mi_mpr_entry_exit_points.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_canvas.h"
#include "MedImgRenderAlgorithm/mi_ray_caster.h"
#include "MedImgRenderAlgorithm/mi_camera_interactor.h"

#include "gl/freeglut.h"


using namespace MED_IMAGING_NAMESPACE;

namespace
{
    std::shared_ptr<ImageDataHeader> m_pDataHeader;
    std::shared_ptr<ImageData> m_pImgData;
    std::shared_ptr<OrthoCamera> m_pCamera;
    std::shared_ptr<OrthoCameraInteractor> m_pCameraInteractor;
    std::shared_ptr<CameraCalculator> m_pCameraCal;
    std::shared_ptr<MPREntryExitPoints> m_pMPREE;
    std::shared_ptr<RayCaster> m_pRayCaster;
    std::shared_ptr<RayCasterCanvas> m_pCanvas;

    int m_iWidth = 800;
    int m_iHeight = 800;
    int m_iButton = -1;
    Point2 m_ptPre;
    int m_iTestCode = 0;

    std::vector<std::string> GetFiles()
    {
        const std::string sFile = "E:/Data/MyData/AB_CTA_01/";
        unsigned int uiSliceCount = 734;
        const std::string sPrefix ="DICOM7_000";
        std::string sCurFile;
        std::vector<std::string> vecFiles;
        for (unsigned int i = 0 ; i< uiSliceCount ; ++i)
        {
            std::stringstream ss;
            if (i<10)
            {
                ss << sFile << sPrefix << "00" << i;
            }
            else if (i<100)
            {
                ss << sFile << sPrefix << "0" << i;
            }
            else
            {
                ss << sFile << sPrefix  << i;
            }
            vecFiles.push_back(ss.str());
        }

        return vecFiles;
    }

    void Init()
    {
        Configuration::instance()->set_processing_unit_type(CPU);
        GLUtils::set_check_gl_flag(true);

        std::vector<std::string> vecFiles = GetFiles();
        DICOMLoader loader;
        IOStatus status = loader.load_series(vecFiles , m_pImgData , m_pDataHeader);

        m_pCamera.reset(new OrthoCamera());
        m_pCameraCal.reset(new CameraCalculator(m_pImgData));
        m_pCameraCal->init_mpr_placement(m_pCamera , TRANSVERSE , Point3(0,0,0));

        m_pCameraInteractor.reset(new OrthoCameraInteractor(m_pCamera));

        m_pMPREE.reset(new MPREntryExitPoints());
        m_pMPREE->set_display_size(m_iWidth,m_iHeight);
        m_pMPREE->set_camera(m_pCamera);
        m_pMPREE->set_camera_calculator(m_pCameraCal);
        m_pMPREE->set_strategy(CPU_BASE);
        m_pMPREE->set_image_data(m_pImgData);
        m_pMPREE->set_thickness(1.0f);

        m_pCanvas.reset(new RayCasterCanvas());
        m_pCanvas->set_display_size(m_iWidth , m_iHeight);
        m_pCanvas->initialize();

        m_pRayCaster.reset(new RayCaster());
        m_pRayCaster->set_entry_exit_points(m_pMPREE);
        m_pRayCaster->set_canvas(m_pCanvas);
        m_pRayCaster->set_camera(m_pCamera);
        m_pRayCaster->set_volume_data(m_pImgData);
        m_pRayCaster->set_volume_to_world_matrix(m_pCameraCal->get_volume_to_world_matrix());
        m_pRayCaster->set_sample_rate(1.0);
        m_pRayCaster->set_global_window_level(252,40+1024);
        m_pRayCaster->set_strategy(CPU_BASE);
        m_pRayCaster->set_composite_mode(COMPOSITE_AVERAGE);


        //Concurrency::instance()->set_app_concurrency(4);
    }

    void RayCasterCanvasToScreen()
    {
        glBindFramebuffer(GL_READ_FRAMEBUFFER , m_pCanvas->get_fbo()->get_id());
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER , 0);
        glDrawBuffer(GL_BACK);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glBlitFramebuffer(0,0,m_iWidth, m_iHeight , 0,0,m_iWidth , m_iHeight , GL_COLOR_BUFFER_BIT , GL_LINEAR);


        /* glEnable(GL_TEXTURE_2D);
        m_pCanvas->get_color_attach_texture()->bind();

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

        m_pCanvas->debug_output_color("D:/mpr_rgba.raw");*/
    }

    void Display()
    {
        glViewport(0,0,m_iWidth , m_iHeight);
        glClearColor(1.0,0.0,0.0,1.0);
        glClear(GL_COLOR_BUFFER_BIT);

        m_pMPREE->calculate_entry_exit_points();
        m_pRayCaster->render(m_iTestCode);


        RayCasterCanvasToScreen();

        //glDrawPixels(m_iWidth , m_iHeight , GL_RGBA , GL_UNSIGNED_BYTE , (void*)m_pCanvas->get_color_array());

        glutSwapBuffers();
    }

    void Keyboard(unsigned char key , int x , int y)
    {
        switch(key)
        {
        case 't':
            {
                std::cout << "W H :" << m_iWidth << " " << m_iHeight << std::endl;
                m_pMPREE->debug_output_entry_points("D:/entry_exit.rgb.raw");
                break;
            }
        case 'a':
            {
                m_pCameraCal->init_mpr_placement(m_pCamera , TRANSVERSE , Point3(0,0,0));
                m_pCameraInteractor->SetInitialStatus(m_pCamera);
                m_pCameraInteractor->Resize(m_iWidth , m_iHeight);
                break;
            }
        case 's':
            {
                m_pCameraCal->init_mpr_placement(m_pCamera , SAGITTAL , Point3(0,0,0));
                m_pCameraInteractor->SetInitialStatus(m_pCamera);
                m_pCameraInteractor->Resize(m_iWidth , m_iHeight);
                break;
            }
        case 'c':
            {
                m_pCameraCal->init_mpr_placement(m_pCamera , CORONAL, Point3(0,0,0));
                m_pCameraInteractor->SetInitialStatus(m_pCamera);
                m_pCameraInteractor->Resize(m_iWidth , m_iHeight);
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

    void Resize(int x , int y)
    {
        m_iWidth = x;
        m_iHeight = y;
        m_pMPREE->set_display_size(m_iWidth , m_iHeight);
        m_pCanvas->set_display_size(m_iWidth , m_iHeight);
        m_pCanvas->update_fbo();
        m_pCameraInteractor->Resize(m_iWidth , m_iHeight);
        glutPostRedisplay();
    }

    void Idle()
    {
        glutPostRedisplay();
    }

    void MouseClick(int button , int status , int x , int y)
    {
        x = x< 0 ? 0 : x;
        x = x> m_iWidth-1 ?  m_iWidth-1 : x;
        y = y< 0 ? 0 : y;
        y = y> m_iHeight-1 ?  m_iHeight-1 : y;

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
        x = x> m_iWidth-1 ?  m_iWidth-1 : x;
        y = y< 0 ? 0 : y;
        y = y> m_iHeight-1 ?  m_iHeight-1 : y;

        Point2 ptCur(x,y);

        //std::cout << "Pre : " << m_ptPre.x << " " <<m_ptPre.y << std::endl;
        //std::cout << "Cur : " << ptCur.x << " " <<ptCur.y << std::endl;
        if (m_iButton == GLUT_LEFT_BUTTON)
        {
            m_pCameraInteractor->rotate(m_ptPre , ptCur , m_iWidth , m_iHeight);
            
        }
        else if (m_iButton == GLUT_MIDDLE_BUTTON)
        {
            m_pCameraInteractor->pan(m_ptPre , ptCur , m_iWidth , m_iHeight);
        }
        else if (m_iButton == GLUT_RIGHT_BUTTON)
        {
            m_pCameraInteractor->zoom(m_ptPre , ptCur , m_iWidth , m_iHeight);
        }

        m_ptPre = ptCur;
        glutPostRedisplay();

    }
}

void UT_CPUMPR(int argc , char* argv[])
{
    try
    {
        glutInit(&argc , argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
        glutInitWindowPosition(0,0);
        glutInitWindowSize(m_iWidth,m_iHeight);

        glutCreateWindow("Test GL resource");

        if ( GLEW_OK != glewInit())
        {
            std::cout <<"Init glew failed!\n";
            return;
        }

        GLEnvironment env;
        int iMajor , iMinor;
        env.get_gl_version(iMajor , iMinor);

        glutDisplayFunc(Display);
        glutReshapeFunc(Resize);
        glutIdleFunc(Idle);
        glutKeyboardFunc(Keyboard);
        glutMouseFunc(MouseClick);
        glutMotionFunc(MouseMotion);

        Init();

        glutMainLoop(); 
    }
    catch (const Exception& e)
    {
        std::cout  << e.what();
        return;
    }
}