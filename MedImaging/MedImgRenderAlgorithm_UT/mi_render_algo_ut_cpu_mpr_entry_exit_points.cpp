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

    int m_iWidth = 512;
    int m_iHeight = 512;
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
        IOStatus status = loader.LoadSeries(vecFiles , m_pImgData , m_pDataHeader);

        m_pCamera.reset(new OrthoCamera());
        m_pCameraCal.reset(new CameraCalculator(m_pImgData));
        m_pCameraCal->InitMPRPlacement(m_pCamera , TRANSVERSE , Point3(0,0,0));

        m_pCameraInteractor.reset(new OrthoCameraInteractor(m_pCamera));

        m_pMPREE.reset(new MPREntryExitPoints());
        m_pMPREE->SetDisplaySize(m_iWidth,m_iHeight);
        m_pMPREE->SetCamera(m_pCamera);
        m_pMPREE->SetCameraCalculator(m_pCameraCal);
        m_pMPREE->SetStrategy(CPU_BASE);
        m_pMPREE->SetImageData(m_pImgData);
        m_pMPREE->SetThickness(1.0f);

        m_pCanvas.reset(new RayCasterCanvas());
        m_pCanvas->SetDisplaySize(m_iWidth , m_iHeight);
        m_pCanvas->Initialize();

        m_pRayCaster.reset(new RayCaster());
        m_pRayCaster->SetEntryExitPoints(m_pMPREE);
        m_pRayCaster->SetCanvas(m_pCanvas);
        m_pRayCaster->SetCamera(m_pCamera);
        m_pRayCaster->SetVolumeData(m_pImgData);
        m_pRayCaster->SetSampleRate(1.0);
        m_pRayCaster->SetGlobalWindowLevel(200,1044);
        m_pRayCaster->SetStrategy(CPU_BASE);
        m_pRayCaster->SetCompositeMode(COMPOSITE_AVERAGE);


        Concurrency::Instance()->SetAppConcurrency(4);
    }

    void RayCasterCanvasToScreen()
    {
        //glViewport(0,0,m_iWidth, m_iHeight);
        glBindFramebuffer(GL_READ_FRAMEBUFFER , m_pCanvas->GetFBO()->GetID());
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER , GL_BACK);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glBlitFramebuffer(0,0,m_iWidth, m_iHeight , 0,0,m_iWidth , m_iHeight , GL_COLOR_BUFFER_BIT , GL_LINEAR);

    }

    void Display()
    {
        glViewport(0,0,m_iWidth , m_iHeight);
        glClearColor(1.0,0.0,0.0,1.0);
        glClear(GL_COLOR_BUFFER_BIT);

        m_pMPREE->CalculateEntryExitPoints();
        m_pRayCaster->Render(1);
        RayCasterCanvasToScreen();


        glutSwapBuffers();
    }

    void Keyboard(unsigned char key , int x , int y)
    {
        switch(key)
        {
        case 't':
            {
                std::cout << "W H :" << m_iWidth << " " << m_iHeight << std::endl;
                m_pMPREE->DebugOutputExitPoints("D:/entry_exit.rgb.raw");
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
        m_pMPREE->SetDisplaySize(m_iWidth , m_iHeight);
        m_pCanvas->SetDisplaySize(m_iWidth , m_iHeight);
        m_pCanvas->UpdateFBO();
        m_pCameraInteractor->Resize(m_iWidth , m_iHeight);
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
            m_pCameraInteractor->Rotate(m_ptPre , ptCur , m_iWidth , m_iHeight);
        }
        else if (m_iButton == GLUT_MIDDLE_BUTTON)
        {
            m_pCameraInteractor->Pan(m_ptPre , ptCur , m_iWidth , m_iHeight);
        }
        else if (m_iButton == GLUT_RIGHT_BUTTON)
        {
            m_pCameraInteractor->Zoom(m_ptPre , ptCur , m_iWidth , m_iHeight);
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
        glutInitWindowSize(m_iWidth,m_iHeight);

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