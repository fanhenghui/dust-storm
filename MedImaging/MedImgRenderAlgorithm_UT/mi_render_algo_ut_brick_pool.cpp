#include "mi_render_algo_ut_stdafx.h"

#include "gl/glew.h"

#include "MedImgCommon/mi_concurrency.h"

#include "MedImgIO/mi_dicom_loader.h"
#include "MedImgIO/mi_image_data.h"
#include "MedImgIO/mi_image_data_header.h"

#include "MedImgArithmetic/mi_ortho_camera.h"

#include "MedImgGLResource/mi_gl_fbo.h"
#include "MedImgGLResource/mi_gl_texture_2d.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "MedImgRenderAlgorithm/mi_camera_calculator.h"
#include "MedImgRenderAlgorithm/mi_mpr_entry_exit_points.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_canvas.h"
#include "MedImgRenderAlgorithm/mi_ray_caster.h"
#include "MedImgRenderAlgorithm/mi_camera_interactor.h"
#include "MedImgRenderAlgorithm/mi_brick_define.h"
#include "MedImgRenderAlgorithm/mi_brick_pool.h"
#include "MedImgRenderAlgorithm/mi_brick_generator.h"
#include "MedImgRenderAlgorithm/mi_brick_info_generator.h"
#include "MedImgRenderAlgorithm/mi_brick_utils.h"

#include "gl/freeglut.h"



using namespace MedImaging;

namespace
{
    //void TestLabelKey()
    //{
    //    std::map<LabelKey , int> mapBrick;
    //    std::vector<unsigned char> vecLabels;

    //    mapBrick[LabelKey(vecLabels)] = 1;

    //    vecLabels.clear();
    //    vecLabels.push_back(1);
    //    mapBrick[LabelKey(vecLabels)] = 2;

    //    vecLabels.clear();
    //    vecLabels.push_back(1);
    //    vecLabels.push_back(3);
    //    vecLabels.push_back(2);
    //    mapBrick[LabelKey(vecLabels)] = 3;


    //    vecLabels.clear();
    //    vecLabels.push_back(5);
    //    vecLabels.push_back(23);
    //    vecLabels.push_back(2);
    //    vecLabels.push_back(9);
    //    mapBrick[LabelKey(vecLabels)] = 4;

    //    for (auto it = mapBrick.begin() ; it != mapBrick.end() ; ++it)
    //    {
    //        std::vector<unsigned char> v = it->first.ExtractLabels();
    //        for (auto it2= v.begin() ; it2 != v.end() ; ++it2)
    //        {
    //            std::cout << (int)(*it2) << " , ";
    //        }
    //        std::cout << "\n";
    //    }

    //    std::cout << "Done\n";
    //}

    std::shared_ptr<ImageDataHeader> m_pDataHeader;
    std::shared_ptr<ImageData> m_pImgData;
    std::shared_ptr<OrthoCamera> m_pCamera;
    std::shared_ptr<OrthoCameraInteractor> m_pCameraInteractor;
    std::shared_ptr<CameraCalculator> m_pCameraCal;
    std::shared_ptr<MPREntryExitPoints> m_pMPREE;
    std::shared_ptr<RayCaster> m_pRayCaster;
    std::shared_ptr<RayCasterCanvas> m_pCanvas;
    std::shared_ptr<BrickPool> m_pBrickPool;

    std::shared_ptr<OrthoCamera> m_pCameraBrick;
    std::shared_ptr<OrthoCameraInteractor> m_pCameraInteractorBrick;
    RayCastingStrategy m_eStrategy = CPU_BRICK_ACCELERATE;

    //WL ��λ������ֵһ����δ����У׼
    float m_fWW = 252;
    float m_fWL = 1064;

    float m_fThickness =0.5;
#ifdef _DEBUG
    int m_iWidth = 512;
    int m_iHeight = 512;
#else
    int m_iWidth = 1024;
    int m_iHeight = 1024;
#endif
    
    int m_iButton = -1;
    Point2 m_ptPre;
    int m_iTestCode = 0;
    unsigned int m_uiBrickTestNum = 100;
    unsigned int m_uiBrickMode = 0;

    std::vector<std::string> GetFiles()
    {
        const std::string sFile = "D:/Data/AB_CTA_01/";
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

    void CalculateBrickPool()
    {
        BrickUtils::Instance()->SetBrickSize(32);
        BrickUtils::Instance()->SetBrickExpand(2);

        m_pBrickPool.reset(new BrickPool());
        m_pBrickPool->SetVolume(m_pImgData);
        

        m_pBrickPool->CalculateVolumeBrick();
    }

    void Init()
    {
        std::vector<std::string> vecFiles = GetFiles();
        DICOMLoader loader;
        IOStatus status = loader.LoadSeries(vecFiles , m_pImgData , m_pDataHeader);

        m_pCamera.reset(new OrthoCamera());
        m_pCameraBrick.reset(new OrthoCamera());
        m_pCameraCal.reset(new CameraCalculator(m_pImgData));
        m_pCameraCal->InitMPRPlacement(m_pCamera , TRANSVERSE , Point3(0,0,0));
        m_pCameraCal->InitVRRPlacement(m_pCameraBrick);
        m_pCameraInteractorBrick.reset(new OrthoCameraInteractor(m_pCameraBrick));
        m_pCameraInteractor.reset(new OrthoCameraInteractor(m_pCamera));

        CalculateBrickPool();

        m_pMPREE.reset(new MPREntryExitPoints());
        m_pMPREE->SetDisplaySize(m_iWidth,m_iHeight);
        m_pMPREE->SetCamera(m_pCamera);
        m_pMPREE->SetCameraCalculator(m_pCameraCal);
        m_pMPREE->SetStrategy(m_eStrategy);
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
        m_pRayCaster->SetGlobalWindowLevel(m_fWW,m_fWL);
        m_pRayCaster->SetStrategy(m_eStrategy);
        m_pRayCaster->SetCompositeMode(COMPOSITE_AVERAGE);

        m_pRayCaster->SetBrickSize(BrickUtils::Instance()->GetBrickSize());
        m_pRayCaster->SetBrickExpand(BrickUtils::Instance()->GetBrickExpand());
        m_pRayCaster->SetBrickCorner(m_pBrickPool->GetBrickCorner());
        m_pRayCaster->SetVolumeBrickUnit(m_pBrickPool->GetVolumeBrickUnit());
        m_pRayCaster->SetVolumeBrickInfo(m_pBrickPool->GetVolumeBrickInfo());

        //Concurrency::Instance()->SetAppConcurrency(4);
    }

    void RayCasterCanvasToScreen()
    {
        glBindFramebuffer(GL_READ_FRAMEBUFFER , m_pCanvas->GetFBO()->GetID());
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER , 0);
        glDrawBuffer(GL_BACK);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glBlitFramebuffer(0,0,m_iWidth, m_iHeight , 0,0,m_iWidth , m_iHeight , GL_COLOR_BUFFER_BIT , GL_LINEAR);
    }

#define PER_VERTEX( x ,  y ,  z) \
    glColor3d(x/fDim[0], y/fDim[1] , z/fDim[2]);\
    glVertex4d(x,y,z,1.0);

    void DrawSingleBrick(Point3 ptMin , Point3 ptMax)
    {

        float fDim[3] = {(float)m_pImgData->m_uiDim[0] , (float)m_pImgData->m_uiDim[1] , (float)m_pImgData->m_uiDim[2]};

        glBegin(GL_QUADS);

        //Z-
        PER_VERTEX(ptMin.x , ptMin.y, ptMin.z);
        PER_VERTEX(ptMax.x , ptMin.y, ptMin.z);
        PER_VERTEX(ptMax.x , ptMax.y, ptMin.z);
        PER_VERTEX(ptMin.x , ptMax.y, ptMin.z);

        //Z +
        PER_VERTEX(ptMin.x , ptMin.y, ptMax.z);
        PER_VERTEX(ptMax.x , ptMin.y, ptMax.z);
        PER_VERTEX(ptMax.x , ptMax.y, ptMax.z);
        PER_VERTEX(ptMin.x , ptMax.y, ptMax.z);

        //X-
        PER_VERTEX(ptMin.x , ptMin.y, ptMin.z);
        PER_VERTEX(ptMin.x , ptMax.y, ptMin.z);
        PER_VERTEX(ptMin.x , ptMax.y, ptMax.z);
        PER_VERTEX(ptMin.x , ptMin.y, ptMax.z);

        //X+
        PER_VERTEX(ptMax.x , ptMin.y, ptMin.z);
        PER_VERTEX(ptMax.x , ptMax.y, ptMin.z);
        PER_VERTEX(ptMax.x , ptMax.y, ptMax.z);
        PER_VERTEX(ptMax.x , ptMin.y, ptMax.z);

        //Y-
        PER_VERTEX(ptMin.x , ptMin.y, ptMin.z);
        PER_VERTEX(ptMin.x , ptMin.y, ptMax.z);
        PER_VERTEX(ptMax.x , ptMin.y, ptMax.z);
        PER_VERTEX(ptMax.x , ptMin.y, ptMin.z);

        //Y+
        PER_VERTEX(ptMin.x , ptMax.y, ptMin.z);
        PER_VERTEX(ptMin.x , ptMax.y, ptMax.z);
        PER_VERTEX(ptMax.x , ptMax.y, ptMax.z);
        PER_VERTEX(ptMax.x , ptMax.y, ptMin.z);

        glEnd();
    }

    void DrawBricks()
    {
        BrickCorner* pBrickCorner = m_pBrickPool->GetBrickCorner();
        VolumeBrickInfo* pBrickInfo = m_pBrickPool->GetVolumeBrickInfo();

        unsigned int uiBrickDim[3];
        m_pBrickPool->GetBrickDim(uiBrickDim);
        const unsigned int uiBrickCount = uiBrickDim[0]*uiBrickDim[1]*uiBrickDim[2];
        const unsigned int uiBrickSize = BrickUtils::Instance()->GetBrickSize();

        //////////////////////////////////////////////////////////////////////////
        //Rendering bricks
        /////////////////////////////////////////////////////////////////////////
        //Render by sorting bricks
        if (m_uiBrickMode == 0)
        {
           
            const std::vector<BrickDistance>& vecBrickDistance = m_pRayCaster->GetBrickDistance();
            const unsigned int uiNum = m_pRayCaster->GetRayCastingBrickCount();
            for (unsigned int i = 0 ; i < uiNum ; ++i)
            {
                unsigned int idx = vecBrickDistance[i].m_id;

                Point3 ptMin( pBrickCorner[idx].m_Min[0] , pBrickCorner[idx].m_Min[1] , pBrickCorner[idx].m_Min[2]);
                Point3 ptMax = ptMin + Vector3(uiBrickSize , uiBrickSize ,uiBrickSize);
                DrawSingleBrick(ptMin, ptMax);
            }
        }
         //Render by WL
        else if (m_uiBrickMode == 1)
        {
           
            for (unsigned int i =0 ; i< uiBrickCount ; ++i)
            {
                //Brick skip by WL
                float fMinGray = m_fWL - m_fWW*2.0f;
                if (pBrickInfo[i].m_fMax < fMinGray)
                {
                    continue;
                }

                Point3 ptMin( pBrickCorner[i].m_Min[0] , pBrickCorner[i].m_Min[1] , pBrickCorner[i].m_Min[2]);
                Point3 ptMax = ptMin + Vector3(uiBrickSize , uiBrickSize ,uiBrickSize);
                DrawSingleBrick(ptMin, ptMax);
            }
        }


        //////////////////////////////////////////////////////////////////////////
        

        {
            /*unsigned int uidx = uiBrickDim[2]/2 * uiBrickDim[0]*uiBrickDim[1] +
                uiBrickDim[1]/2 * uiBrickDim[0] +uiBrickDim[1]/2;

            Point3 ptMin( pBrickCorner[uidx].m_Min[0] , pBrickCorner[uidx].m_Min[1] , pBrickCorner[uidx].m_Min[2]);
            Point3 ptMax = ptMin + Vector3(uiBrickSize , uiBrickSize ,uiBrickSize);
            DrawSingleBrick(ptMin, ptMax);*/

            //DrawSingleBrick(Point3(0,0,0), Point3(m_pImgData->m_uiDim[0] , m_pImgData->m_uiDim[1], m_pImgData->m_uiDim[2]));
        }
    }

    void Display()
    {
        CHECK_GL_ERROR

        glViewport(0,0,m_iWidth , m_iHeight);

        glClearColor(0.0,0.0,0.0,0.0);
        glClearDepth(1.0);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        clock_t t0=clock();
        m_pMPREE->CalculateEntryExitPoints();
        m_pRayCaster->Render(m_iTestCode);
        clock_t t1=clock();
        std::cout << "Render cost : " << double(t1 - t0) << std::endl;
       /* glPushAttrib(GL_ALL_ATTRIB_BITS);

        glPushMatrix();
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glLoadMatrixd(m_pCameraBrick->GetViewProjectionMatrix()._m);
        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixd(m_pCameraCal->GetVolumeToWorldMatrix()._m);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);

        glPolygonMode(GL_FRONT_AND_BACK , GL_FILL);
        DrawBricks();

        glPopMatrix();
        glPopAttrib();*/

        CHECK_GL_ERROR

        



        RayCasterCanvasToScreen();

        //glDrawPixels(m_iWidth , m_iHeight , GL_RGBA , GL_UNSIGNED_BYTE , (void*)m_pCanvas->GetColorArray());

        glutSwapBuffers();
    }

    void Keyboard(unsigned char key , int x , int y)
    {
        switch(key)
        {
        case 't':
            {
                std::cout << "W H :" << m_iWidth << " " << m_iHeight << std::endl;
                m_pMPREE->DebugOutputEntryPoints("D:/temp/entry_points.rgb.raw");
                m_pMPREE->DebugOutputExitPoints("D:/temp/exit_points.rgb.raw");
                m_pCanvas->DebugOutputColor("D:/temp/out_put_rgba.raw");
                const std::vector<BrickDistance>& vecBrickDis = m_pRayCaster->GetBrickDistance();
                unsigned int uiBrickNum = m_pRayCaster->GetRayCastingBrickCount();
                std::ofstream out("D:/temp/brick_sort.txt" , std::ios::out);
                if (out.is_open())
                {
                    out << "Brick number : " << uiBrickNum  << std::endl;
                    for (unsigned int i = 0 ; i<uiBrickNum ; ++i)
                    {
                        out << vecBrickDis[i].m_id << std::endl;
                    }
                    out.close();
                }
                break;
            }
        case 'g':
            {
                if (m_eStrategy == CPU_BASE)
                {
                    m_eStrategy = CPU_BRICK_ACCELERATE;
                    std::cout << "Ray casting strategy is : CPU brick acceleration. \n";
                }
                else
                {
                    m_eStrategy = CPU_BASE;
                    std::cout << "Ray casting strategy is : CPU based. \n";
                }
                m_pRayCaster->SetStrategy(m_eStrategy);
                m_pMPREE->SetStrategy(m_eStrategy);
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
        case '8':
            {
                //m_uiBrickTestNum +=10;
                m_pCameraCal->MPROrthoPaging(m_pCamera, -16);
                break;
            }
        case '2':
            {
                //m_uiBrickTestNum -=10;
                m_pCameraCal->MPROrthoPaging(m_pCamera, 16);
                break;
            }

        case '1':
            {
                m_uiBrickMode = 1;
                break;
            }
        case '0':
            {
                m_uiBrickMode = 0;
                break;
            }
        case '6':
            {
                m_fThickness+=1.0;
                std::cout << "Thickness : " << m_fThickness << std::endl;
                m_pMPREE->SetThickness(m_fThickness);
                break;
            }
        case '4':
            {
                m_fThickness-=1.0;
                if (m_fThickness < 1.0f)
                {
                    m_fThickness = 0.5f;
                }
                std::cout << "Thickness : " << m_fThickness << std::endl;
                m_pMPREE->SetThickness(m_fThickness);
                break;
            }
        case 'h':
            {
                if (m_fThickness < 5.0f)
                {
                    m_fThickness = 30.0f;
                }
                else
                {
                    m_fThickness  = 1.0f;
                }
                m_pMPREE->SetThickness(m_fThickness);
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
        m_pCameraInteractorBrick->Resize(m_iWidth , m_iHeight);
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
            //m_pCameraInteractorBrick->Rotate(m_ptPre , ptCur , m_iWidth , m_iHeight);
            m_pCameraInteractor->Rotate(m_ptPre , ptCur , m_iWidth , m_iHeight);
        }
        else if (m_iButton == GLUT_MIDDLE_BUTTON)
        {
            m_pCameraInteractor->Pan(m_ptPre , ptCur , m_iWidth , m_iHeight);
            /*float fDeltaX = x - m_ptPre.x;
            float fDeltaY = m_ptPre.y - y;
            if (m_fWW + fDeltaX > 1.0f)
            {
            m_fWW += fDeltaX;
            }
            m_fWL += fDeltaY;
            m_pRayCaster->SetGlobalWindowLevel(m_fWW, m_fWL);*/
        }
        else if (m_iButton == GLUT_RIGHT_BUTTON)
        {
            //m_pCameraInteractorBrick->Zoom(m_ptPre , ptCur , m_iWidth , m_iHeight);
            m_pCameraInteractor->Zoom(m_ptPre , ptCur , m_iWidth , m_iHeight);
        }

        m_ptPre = ptCur;
        glutPostRedisplay();

    }


}

void UT_BrickPool(int argc , char* argv[])
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

        //return;

        glutMainLoop(); 
    }
    catch (const Exception& e)
    {
        std::cout  << e.what();
        return;
    }
}