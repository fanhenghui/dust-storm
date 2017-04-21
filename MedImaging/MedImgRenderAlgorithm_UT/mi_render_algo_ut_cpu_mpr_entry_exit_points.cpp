#include "gl/glew.h"

#include "MedImgCommon/mi_concurrency.h"

#include "MedImgIO/mi_dicom_loader.h"
#include "MedImgIO/mi_image_data.h"
#include "MedImgIO/mi_image_data_header.h"

#include "MedImgArithmetic/mi_ortho_camera.h"

#include "MedImgGLResource/mi_gl_fbo.h"

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

    int _width = 512;
    int _height = 512;
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
        std::vector<std::string> vecFiles = GetFiles();
        DICOMLoader loader;
        IOStatus status = loader.load_series(vecFiles , m_pImgData , m_pDataHeader);

        m_pCamera.reset(new OrthoCamera());
        m_pCameraCal.reset(new CameraCalculator(m_pImgData));
        m_pCameraCal->init_mpr_placement(m_pCamera , TRANSVERSE , Point3(0,0,0));

        m_pCameraInteractor.reset(new OrthoCameraInteractor(m_pCamera));

        m_pMPREE.reset(new MPREntryExitPoints());
        m_pMPREE->set_display_size(_width,_height);
        m_pMPREE->set_camera(m_pCamera);
        m_pMPREE->set_camera_calculator(m_pCameraCal);
        m_pMPREE->set_strategy(CPU_BASE);
        m_pMPREE->set_image_data(m_pImgData);
        m_pMPREE->set_thickness(1.0f);

        m_pCanvas.reset(new RayCasterCanvas());
        m_pCanvas->set_display_size(_width , _height);
        m_pCanvas->initialize();

        m_pRayCaster.reset(new RayCaster());
        m_pRayCaster->set_entry_exit_points(m_pMPREE);
        m_pRayCaster->set_canvas(m_pCanvas);
        m_pRayCaster->set_camera(m_pCamera);
        m_pRayCaster->set_volume_data(m_pImgData);
        m_pRayCaster->set_sample_rate(1.0);
        m_pRayCaster->set_global_window_level(200,1044);
        m_pRayCaster->set_strategy(CPU_BASE);
        m_pRayCaster->set_composite_mode(COMPOSITE_AVERAGE);


        Concurrency::instance()->set_app_concurrency(4);
    }

    void RayCasterCanvasToScreen()
    {
        //glViewport(0,0,_width, _height);
        glBindFramebuffer(GL_READ_FRAMEBUFFER , m_pCanvas->get_fbo()->get_id());
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER , GL_BACK);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glBlitFramebuffer(0,0,_width, _height , 0,0,_width , _height , GL_COLOR_BUFFER_BIT , GL_LINEAR);

    }

    void Display()
    {
        glViewport(0,0,_width , _height);
        glClearColor(1.0,0.0,0.0,1.0);
        glClear(GL_COLOR_BUFFER_BIT);

        m_pMPREE->calculate_entry_exit_points();
        m_pRayCaster->render(1);
        RayCasterCanvasToScreen();


        glutSwapBuffers();
    }

    void Keyboard(unsigned char key , int x , int y)
    {
        switch(key)
        {
        case 't':
            {
                std::cout << "W H :" << _width << " " << _height << std::endl;
                m_pMPREE->debug_output_exit_points("D:/entry_exit.rgb.raw");
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
        _width = x;
        _height = y;
        m_pMPREE->set_display_size(_width , _height);
        m_pCanvas->set_display_size(_width , _height);
        m_pCanvas->update_fbo();
        m_pCameraInteractor->Resize(_width , _height);
        glutPostRedisplay();
    }

    void Idle()
    {
        //glutPostRedisplay();
    }

    void MouseClick(int button , int status , int x , int y)
    {
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
        Point2 ptCur(x,y);
        if (m_iButton == GLUT_LEFT_BUTTON)
        {
            m_pCameraInteractor->rotate(m_ptPre , ptCur , _width , _height);
        }
        else if (m_iButton == GLUT_MIDDLE_BUTTON)
        {
            m_pCameraInteractor->pan(m_ptPre , ptCur , _width , _height);
        }
        else if (m_iButton == GLUT_RIGHT_BUTTON)
        {
            m_pCameraInteractor->zoom(m_ptPre , ptCur , _width , _height);
        }

        m_ptPre = ptCur;
        glutPostRedisplay();

    }
}

void UT_CPUMPREntryExitPoints(int argc , char* argv[])
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
            return;
        }

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