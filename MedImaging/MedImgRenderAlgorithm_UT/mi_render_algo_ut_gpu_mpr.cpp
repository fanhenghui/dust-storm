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
#include "MedImgGLResource/mi_gl_texture_1d.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "MedImgRenderAlgorithm/mi_camera_calculator.h"
#include "MedImgRenderAlgorithm/mi_mpr_entry_exit_points.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_canvas.h"
#include "MedImgRenderAlgorithm/mi_ray_caster.h"
#include "MedImgRenderAlgorithm/mi_camera_interactor.h"
#include "MedImgRenderAlgorithm/mi_volume_infos.h"

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
    std::shared_ptr<VolumeInfos> m_pVolumeInfos;

    std::shared_ptr<GLTexture1D> m_pPseudoColor;

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
        Configuration::Instance()->SetProcessingUnitType(GPU);
        GLUtils::SetCheckGLFlag(true);

        std::vector<std::string> vecFiles = GetFiles();
        DICOMLoader loader;
        IOStatus status = loader.LoadSeries(vecFiles , m_pImgData , m_pDataHeader);

        m_pVolumeInfos.reset( new VolumeInfos());
        m_pVolumeInfos->SetDataHeader(m_pDataHeader);
        m_pVolumeInfos->SetVolume(m_pImgData);

        m_pCamera.reset(new OrthoCamera());
        m_pCameraCal.reset(new CameraCalculator(m_pImgData));
        m_pCameraCal->InitMPRPlacement(m_pCamera , TRANSVERSE , Point3(0,0,0));

        m_pCameraInteractor.reset(new OrthoCameraInteractor(m_pCamera));

        m_pMPREE.reset(new MPREntryExitPoints());
        m_pMPREE->SetDisplaySize(m_iWidth,m_iHeight);
        m_pMPREE->SetCamera(m_pCamera);
        m_pMPREE->SetCameraCalculator(m_pCameraCal);
        m_pMPREE->SetStrategy(GPU_BASE);
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
        m_pRayCaster->SetVolumeToWorldMatrix(m_pCameraCal->GetVolumeToWorldMatrix());
        m_pRayCaster->SetSampleRate(1.0);
        m_pRayCaster->SetGlobalWindowLevel(252,40+1024);
        m_pRayCaster->SetStrategy(GPU_BASE);
        m_pRayCaster->SetCompositeMode(COMPOSITE_AVERAGE);
        m_pRayCaster->SetColorInverseMode(COLOR_INVERSE_DISABLE);

        //GPU
        m_pRayCaster->SetVolumeDataTexture(m_pVolumeInfos->GetVolumeTexture());

        UIDType uid;
        m_pPseudoColor = GLResourceManagerContainer::Instance()->GetTexture1DManager()->CreateObject(uid);
        m_pPseudoColor->SetDescription("Pseudo color texture gray");
        m_pPseudoColor->Initialize();
        m_pPseudoColor->Bind();
        GLTextureUtils::Set1DWrapS(GL_CLAMP_TO_EDGE);
        GLTextureUtils::SetFilter(GL_TEXTURE_1D , GL_LINEAR);
        unsigned char pData[] = {0,0,0,0,255,255,255,255};
        m_pPseudoColor->Load(GL_RGBA8 , 2, GL_RGBA , GL_UNSIGNED_BYTE , pData);

        m_pRayCaster->SetPseudoColorTexture(m_pPseudoColor , 2);

    }

    void RayCasterCanvasToScreen()
    {
        glBindFramebuffer(GL_READ_FRAMEBUFFER , m_pCanvas->GetFBO()->GetID());
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER , 0);
        glDrawBuffer(GL_BACK);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glBlitFramebuffer(0,0,m_iWidth, m_iHeight , 0,0,m_iWidth , m_iHeight , GL_COLOR_BUFFER_BIT , GL_LINEAR);


        /* glEnable(GL_TEXTURE_2D);
        m_pCanvas->GetColorAttachTexture()->Bind();

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

        m_pCanvas->DebugOutputColor("D:/mpr_rgba.raw");*/
    }

    void Display()
    {
        try
        {
            glViewport(0,0,m_iWidth , m_iHeight);
            glClearColor(1.0,0.0,0.0,1.0);
            glClear(GL_COLOR_BUFFER_BIT);

            m_pMPREE->CalculateEntryExitPoints();
            m_pRayCaster->Render(m_iTestCode);


            RayCasterCanvasToScreen();

            //glDrawPixels(m_iWidth , m_iHeight , GL_RGBA , GL_UNSIGNED_BYTE , (void*)m_pCanvas->GetColorArray());

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
                std::cout << "W H :" << m_iWidth << " " << m_iHeight << std::endl;
                m_pMPREE->DebugOutputEntryPoints("D:/entry_exit.rgb.raw");
                break;
            }
        case 'a':
            {
                m_pCameraCal->InitMPRPlacement(m_pCamera , TRANSVERSE , Point3(0,0,0));
                m_pCameraInteractor->SetInitialStatus(m_pCamera);
                m_pCameraInteractor->Resize(m_iWidth , m_iHeight);
                break;
            }
        case 's':
            {
                m_pCameraCal->InitMPRPlacement(m_pCamera , SAGITTAL , Point3(0,0,0));
                m_pCameraInteractor->SetInitialStatus(m_pCamera);
                m_pCameraInteractor->Resize(m_iWidth , m_iHeight);
                break;
            }
        case 'c':
            {
                m_pCameraCal->InitMPRPlacement(m_pCamera , CORONAL, Point3(0,0,0));
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
        m_pMPREE->SetDisplaySize(m_iWidth , m_iHeight);
        m_pCanvas->SetDisplaySize(m_iWidth , m_iHeight);
        m_pCanvas->UpdateFBO();
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

void UT_GPUMPR(int argc , char* argv[])
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
        env.GetGLVersion(iMajor , iMinor);

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