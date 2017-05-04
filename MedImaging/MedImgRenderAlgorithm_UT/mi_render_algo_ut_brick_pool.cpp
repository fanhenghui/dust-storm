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

    std::shared_ptr<ImageDataHeader> _data_header;
    std::shared_ptr<ImageData> _volume_data;
    std::shared_ptr<OrthoCamera> _camera;
    std::shared_ptr<OrthoCameraInteractor> m_pCameraInteractor;
    std::shared_ptr<CameraCalculator> m_pCameraCal;
    std::shared_ptr<MPREntryExitPoints> m_pMPREE;
    std::shared_ptr<RayCaster> _ray_caster;
    std::shared_ptr<RayCasterCanvas> _canvas;
    std::shared_ptr<BrickPool> m_pBrickPool;

    std::shared_ptr<OrthoCamera> m_pCameraBrick;
    std::shared_ptr<OrthoCameraInteractor> m_pCameraInteractorBrick;
    RayCastingStrategy _strategy = CPU_BRICK_ACCELERATE;

    //WL 单位和像素值一样，未经过校准
    float m_fWW = 252;
    float m_fWL = 1064;

    float _thickness =0.5;
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
        m_pBrickPool->set_volume(_volume_data);
        

        m_pBrickPool->calculate_volume_brick();
    }

    void Init()
    {
        std::vector<std::string> files = GetFiles();
        DICOMLoader loader;
        IOStatus status = loader.load_series(files , _volume_data , _data_header);

        _camera.reset(new OrthoCamera());
        m_pCameraBrick.reset(new OrthoCamera());
        m_pCameraCal.reset(new CameraCalculator(_volume_data));
        m_pCameraCal->init_mpr_placement(_camera , TRANSVERSE , Point3(0,0,0));
        m_pCameraCal->init_vr_placement(m_pCameraBrick);
        m_pCameraInteractorBrick.reset(new OrthoCameraInteractor(m_pCameraBrick));
        m_pCameraInteractor.reset(new OrthoCameraInteractor(_camera));

        CalculateBrickPool();

        m_pMPREE.reset(new MPREntryExitPoints());
        m_pMPREE->set_display_size(_width,_height);
        m_pMPREE->set_camera(_camera);
        m_pMPREE->set_camera_calculator(m_pCameraCal);
        m_pMPREE->set_strategy(_strategy);
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
        _ray_caster->set_global_window_level(m_fWW,m_fWL);
        _ray_caster->set_strategy(_strategy);
        _ray_caster->set_composite_mode(COMPOSITE_AVERAGE);

        _ray_caster->set_brick_size(BrickUtils::instance()->GetBrickSize());
        _ray_caster->set_brick_expand(BrickUtils::instance()->get_brick_expand());
        _ray_caster->set_brick_corner(m_pBrickPool->get_brick_corner());
        _ray_caster->set_volume_brick_unit(m_pBrickPool->get_volume_brick_unit());
        _ray_caster->set_volume_brick_info(m_pBrickPool->get_volume_brick_info());

        //Concurrency::instance()->set_app_concurrency(4);
    }

    void RayCasterCanvasToScreen()
    {
        glBindFramebuffer(GL_READ_FRAMEBUFFER , _canvas->get_fbo()->get_id());
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

        float fDim[3] = {(float)_volume_data->_dim[0] , (float)_volume_data->_dim[1] , (float)_volume_data->_dim[2]};

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
        BrickCorner* _brick_corner_array = m_pBrickPool->get_brick_corner();
        VolumeBrickInfo* brick_info_array = m_pBrickPool->get_volume_brick_info();

        unsigned int brick_dim[3];
        m_pBrickPool->get_brick_dim(brick_dim);
        const unsigned int brick_count = brick_dim[0]*brick_dim[1]*brick_dim[2];
        const unsigned int brick_size = BrickUtils::instance()->GetBrickSize();

        //////////////////////////////////////////////////////////////////////////
        //Rendering bricks
        /////////////////////////////////////////////////////////////////////////
        //render by sorting bricks
        if (m_uiBrickMode == 0)
        {
           
            const std::vector<BrickDistance>& vecBrickDistance = _ray_caster->get_brick_distance();
            const unsigned int uiNum = _ray_caster->get_ray_casting_brick_count();
            for (unsigned int i = 0 ; i < uiNum ; ++i)
            {
                unsigned int idx = vecBrickDistance[i].id;

                Point3 ptMin( _brick_corner_array[idx].min[0] , _brick_corner_array[idx].min[1] , _brick_corner_array[idx].min[2]);
                Point3 ptMax = ptMin + Vector3(brick_size , brick_size ,brick_size);
                DrawSingleBrick(ptMin, ptMax);
            }
        }
         //render by WL
        else if (m_uiBrickMode == 1)
        {
           
            for (unsigned int i =0 ; i< brick_count ; ++i)
            {
                //Brick skip by WL
                float fMinGray = m_fWL - m_fWW*2.0f;
                if (brick_info_array[i].max < fMinGray)
                {
                    continue;
                }

                Point3 ptMin( _brick_corner_array[i].min[0] , _brick_corner_array[i].min[1] , _brick_corner_array[i].min[2]);
                Point3 ptMax = ptMin + Vector3(brick_size , brick_size ,brick_size);
                DrawSingleBrick(ptMin, ptMax);
            }
        }


        //////////////////////////////////////////////////////////////////////////
        

        {
            /*unsigned int uidx = brick_dim[2]/2 * brick_dim[0]*brick_dim[1] +
                brick_dim[1]/2 * brick_dim[0] +brick_dim[1]/2;

            Point3 ptMin( _brick_corner_array[uidx].m_Min[0] , _brick_corner_array[uidx].m_Min[1] , _brick_corner_array[uidx].m_Min[2]);
            Point3 ptMax = ptMin + Vector3(brick_size , brick_size ,brick_size);
            DrawSingleBrick(ptMin, ptMax);*/

            //DrawSingleBrick(Point3(0,0,0), Point3(_volume_data->_dim[0] , _volume_data->_dim[1], _volume_data->_dim[2]));
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
        _ray_caster->render(m_iTestCode);
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

        //glDrawPixels(_width , _height , GL_RGBA , GL_UNSIGNED_BYTE , (void*)_canvas->get_color_array());

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
                _canvas->debug_output_color("D:/temp/out_put_rgba.raw");
                const std::vector<BrickDistance>& vecBrickDis = _ray_caster->get_brick_distance();
                unsigned int uiBrickNum = _ray_caster->get_ray_casting_brick_count();
                std::ofstream out("D:/temp/brick_sort.txt" , std::ios::out);
                if (out.is_open())
                {
                    out << "Brick number : " << uiBrickNum  << std::endl;
                    for (unsigned int i = 0 ; i<uiBrickNum ; ++i)
                    {
                        out << vecBrickDis[i].id << std::endl;
                    }
                    out.close();
                }
                break;
            }
        case 'g':
            {
                if (_strategy == CPU_BASE)
                {
                    _strategy = CPU_BRICK_ACCELERATE;
                    std::cout << "Ray casting strategy is : CPU brick acceleration. \n";
                }
                else
                {
                    _strategy = CPU_BASE;
                    std::cout << "Ray casting strategy is : CPU based. \n";
                }
                _ray_caster->set_strategy(_strategy);
                m_pMPREE->set_strategy(_strategy);
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
        case '8':
            {
                //m_uiBrickTestNum +=10;
                m_pCameraCal->page_orthognal_mpr(_camera, -16);
                break;
            }
        case '2':
            {
                //m_uiBrickTestNum -=10;
                m_pCameraCal->page_orthognal_mpr(_camera, 16);
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
                _thickness+=1.0;
                std::cout << "Thickness : " << _thickness << std::endl;
                m_pMPREE->set_thickness(_thickness);
                break;
            }
        case '4':
            {
                _thickness-=1.0;
                if (_thickness < 1.0f)
                {
                    _thickness = 0.5f;
                }
                std::cout << "Thickness : " << _thickness << std::endl;
                m_pMPREE->set_thickness(_thickness);
                break;
            }
        case 'h':
            {
                if (_thickness < 5.0f)
                {
                    _thickness = 30.0f;
                }
                else
                {
                    _thickness  = 1.0f;
                }
                m_pMPREE->set_thickness(_thickness);
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
        m_pCameraInteractorBrick->resize(_width , _height);
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
            //m_pCameraInteractorBrick->rotate(m_ptPre , cur_pt , _width , _height);
            m_pCameraInteractor->rotate(m_ptPre , cur_pt , _width , _height);
        }
        else if (m_iButton == GLUT_MIDDLE_BUTTON)
        {
            m_pCameraInteractor->pan(m_ptPre , cur_pt , _width , _height);
            /*float fDeltaX = x - m_ptPre.x;
            float fDeltaY = m_ptPre.y - y;
            if (m_fWW + fDeltaX > 1.0f)
            {
            m_fWW += fDeltaX;
            }
            m_fWL += fDeltaY;
            _ray_caster->set_global_window_level(m_fWW, m_fWL);*/
        }
        else if (m_iButton == GLUT_RIGHT_BUTTON)
        {
            //m_pCameraInteractorBrick->zoom(m_ptPre , cur_pt , _width , _height);
            m_pCameraInteractor->zoom(m_ptPre , cur_pt , _width , _height);
        }

        m_ptPre = cur_pt;
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
        glutReshapeFunc(resize);
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