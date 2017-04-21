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



using namespace medical_imaging;

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

    //WL 单位和像素值一样，未经过校准
    float m_fWW = 252;
    float m_fWL = 1064;

    float m_fThickness =0.5;
#ifdef _DEBUG
    int _width = 512;
    int _height = 512;
#else
    int _width = 1024;
    int _height = 1024;
#endif
    
    int m_iButton = -1;
    Point2 m_ptPre;
    int m_iTestCode = 0;
    unsigned int m_uiBrickTestNum = 100;
    unsigned int m_uiBrickMode = 0;

    std::vector<std::string> GetFiles()
    {
        const std::string file_name = "D:/Data/AB_CTA_01/";
        unsigned int uiSliceCount = 734;
        const std::string sPrefix ="DICOM7_000";
        std::string sCurFile;
        std::vector<std::string> files;
        for (unsigned int i = 0 ; i< uiSliceCount ; ++i)
        {
            std::stringstream ss;
            if (i<10)
            {
                ss << file_name << sPrefix << "00" << i;
            }
            else if (i<100)
            {
                ss << file_name << sPrefix << "0" << i;
            }
            else
            {
                ss << file_name << sPrefix  << i;
            }
            files.push_back(ss.str());
        }

        return files;
    }

    void CalculateBrickPool()
    {
        BrickUtils::instance()->set_brick_size(32);
        BrickUtils::instance()->set_brick_expand(2);

        m_pBrickPool.reset(new BrickPool());
        m_pBrickPool->set_volume(m_pImgData);
        

        m_pBrickPool->calculate_volume_brick();
    }

    void Init()
    {
        std::vector<std::string> files = GetFiles();
        DICOMLoader loader;
        IOStatus status = loader.load_series(files , m_pImgData , m_pDataHeader);

        m_pCamera.reset(new OrthoCamera());
        m_pCameraBrick.reset(new OrthoCamera());
        m_pCameraCal.reset(new CameraCalculator(m_pImgData));
        m_pCameraCal->init_mpr_placement(m_pCamera , TRANSVERSE , Point3(0,0,0));
        m_pCameraCal->init_vr_placement(m_pCameraBrick);
        m_pCameraInteractorBrick.reset(new OrthoCameraInteractor(m_pCameraBrick));
        m_pCameraInteractor.reset(new OrthoCameraInteractor(m_pCamera));

        CalculateBrickPool();

        m_pMPREE.reset(new MPREntryExitPoints());
        m_pMPREE->set_display_size(_width,_height);
        m_pMPREE->set_camera(m_pCamera);
        m_pMPREE->set_camera_calculator(m_pCameraCal);
        m_pMPREE->set_strategy(m_eStrategy);
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
        m_pRayCaster->set_volume_to_world_matrix(m_pCameraCal->get_volume_to_world_matrix());
        m_pRayCaster->set_sample_rate(1.0);
        m_pRayCaster->set_global_window_level(m_fWW,m_fWL);
        m_pRayCaster->set_strategy(m_eStrategy);
        m_pRayCaster->set_composite_mode(COMPOSITE_AVERAGE);

        m_pRayCaster->set_brick_size(BrickUtils::instance()->GetBrickSize());
        m_pRayCaster->set_brick_expand(BrickUtils::instance()->get_brick_expand());
        m_pRayCaster->set_brick_corner(m_pBrickPool->get_brick_corner());
        m_pRayCaster->set_volume_brick_unit(m_pBrickPool->get_volume_brick_unit());
        m_pRayCaster->set_volume_brick_info(m_pBrickPool->get_volume_brick_info());

        //Concurrency::instance()->set_app_concurrency(4);
    }

    void RayCasterCanvasToScreen()
    {
        glBindFramebuffer(GL_READ_FRAMEBUFFER , m_pCanvas->get_fbo()->get_id());
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER , 0);
        glDrawBuffer(GL_BACK);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glBlitFramebuffer(0,0,_width, _height , 0,0,_width , _height , GL_COLOR_BUFFER_BIT , GL_LINEAR);
    }

#define PER_VERTEX( x ,  y ,  z) \
    glColor3d(x/fDim[0], y/fDim[1] , z/fDim[2]);\
    glVertex4d(x,y,z,1.0);

    void DrawSingleBrick(Point3 ptMin , Point3 ptMax)
    {

        float fDim[3] = {(float)m_pImgData->_dim[0] , (float)m_pImgData->_dim[1] , (float)m_pImgData->_dim[2]};

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
        BrickCorner* pBrickCorner = m_pBrickPool->get_brick_corner();
        VolumeBrickInfo* pBrickInfo = m_pBrickPool->get_volume_brick_info();

        unsigned int uiBrickDim[3];
        m_pBrickPool->get_brick_dim(uiBrickDim);
        const unsigned int uiBrickCount = uiBrickDim[0]*uiBrickDim[1]*uiBrickDim[2];
        const unsigned int uiBrickSize = BrickUtils::instance()->GetBrickSize();

        //////////////////////////////////////////////////////////////////////////
        //Rendering bricks
        /////////////////////////////////////////////////////////////////////////
        //render by sorting bricks
        if (m_uiBrickMode == 0)
        {
           
            const std::vector<BrickDistance>& vecBrickDistance = m_pRayCaster->get_brick_distance();
            const unsigned int uiNum = m_pRayCaster->get_ray_casting_brick_count();
            for (unsigned int i = 0 ; i < uiNum ; ++i)
            {
                unsigned int idx = vecBrickDistance[i].m_id;

                Point3 ptMin( pBrickCorner[idx].m_Min[0] , pBrickCorner[idx].m_Min[1] , pBrickCorner[idx].m_Min[2]);
                Point3 ptMax = ptMin + Vector3(uiBrickSize , uiBrickSize ,uiBrickSize);
                DrawSingleBrick(ptMin, ptMax);
            }
        }
         //render by WL
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

            //DrawSingleBrick(Point3(0,0,0), Point3(m_pImgData->_dim[0] , m_pImgData->_dim[1], m_pImgData->_dim[2]));
        }
    }

    void Display()
    {
        CHECK_GL_ERROR

        glViewport(0,0,_width , _height);

        glClearColor(0.0,0.0,0.0,0.0);
        glClearDepth(1.0);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        clock_t t0=clock();
        m_pMPREE->calculate_entry_exit_points();
        m_pRayCaster->render(m_iTestCode);
        clock_t t1=clock();
        std::cout << "render cost : " << double(t1 - t0) << std::endl;
       /* glPushAttrib(GL_ALL_ATTRIB_BITS);

        glPushMatrix();
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glLoadMatrixd(m_pCameraBrick->get_view_projection_matrix()._m);
        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixd(m_pCameraCal->get_volume_to_world_matrix()._m);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);

        glPolygonMode(GL_FRONT_AND_BACK , GL_FILL);
        DrawBricks();

        glPopMatrix();
        glPopAttrib();*/

        CHECK_GL_ERROR

        



        RayCasterCanvasToScreen();

        //glDrawPixels(_width , _height , GL_RGBA , GL_UNSIGNED_BYTE , (void*)m_pCanvas->get_color_array());

        glutSwapBuffers();
    }

    void Keyboard(unsigned char key , int x , int y)
    {
        switch(key)
        {
        case 't':
            {
                std::cout << "W H :" << _width << " " << _height << std::endl;
                m_pMPREE->debug_output_entry_points("D:/temp/entry_points.rgb.raw");
                m_pMPREE->debug_output_exit_points("D:/temp/exit_points.rgb.raw");
                m_pCanvas->debug_output_color("D:/temp/out_put_rgba.raw");
                const std::vector<BrickDistance>& vecBrickDis = m_pRayCaster->get_brick_distance();
                unsigned int uiBrickNum = m_pRayCaster->get_ray_casting_brick_count();
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
                m_pRayCaster->set_strategy(m_eStrategy);
                m_pMPREE->set_strategy(m_eStrategy);
                break;
            }
        case 'a':
            {
                m_pCameraCal->init_mpr_placement(m_pCamera , TRANSVERSE , Point3(0,0,0));
                m_pCameraInteractor->SetInitialStatus(m_pCamera);
                m_pCameraInteractor->Resize(_width , _height);
                break;
            }
        case 's':
            {
                m_pCameraCal->init_mpr_placement(m_pCamera , SAGITTAL , Point3(0,0,0));
                m_pCameraInteractor->SetInitialStatus(m_pCamera);
                m_pCameraInteractor->Resize(_width , _height);
                break;
            }
        case 'c':
            {
                m_pCameraCal->init_mpr_placement(m_pCamera , CORONAL, Point3(0,0,0));
                m_pCameraInteractor->SetInitialStatus(m_pCamera);
                m_pCameraInteractor->Resize(_width , _height);
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
                m_pCameraCal->page_orthognal_mpr(m_pCamera, -16);
                break;
            }
        case '2':
            {
                //m_uiBrickTestNum -=10;
                m_pCameraCal->page_orthognal_mpr(m_pCamera, 16);
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
                m_pMPREE->set_thickness(m_fThickness);
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
                m_pMPREE->set_thickness(m_fThickness);
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
                m_pMPREE->set_thickness(m_fThickness);
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
        m_pCameraInteractorBrick->Resize(_width , _height);
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

        Point2 ptCur(x,y);

        //std::cout << "Pre : " << m_ptPre.x << " " <<m_ptPre.y << std::endl;
        //std::cout << "Cur : " << ptCur.x << " " <<ptCur.y << std::endl;
        if (m_iButton == GLUT_LEFT_BUTTON)
        {
            //m_pCameraInteractorBrick->rotate(m_ptPre , ptCur , _width , _height);
            m_pCameraInteractor->rotate(m_ptPre , ptCur , _width , _height);
        }
        else if (m_iButton == GLUT_MIDDLE_BUTTON)
        {
            m_pCameraInteractor->pan(m_ptPre , ptCur , _width , _height);
            /*float fDeltaX = x - m_ptPre.x;
            float fDeltaY = m_ptPre.y - y;
            if (m_fWW + fDeltaX > 1.0f)
            {
            m_fWW += fDeltaX;
            }
            m_fWL += fDeltaY;
            m_pRayCaster->set_global_window_level(m_fWW, m_fWL);*/
        }
        else if (m_iButton == GLUT_RIGHT_BUTTON)
        {
            //m_pCameraInteractorBrick->zoom(m_ptPre , ptCur , _width , _height);
            m_pCameraInteractor->zoom(m_ptPre , ptCur , _width , _height);
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

        //return;

        glutMainLoop(); 
    }
    catch (const Exception& e)
    {
        std::cout  << e.what();
        return;
    }
}