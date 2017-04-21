#include "gl/glew.h"

#include "MedImgArithmetic/mi_ortho_camera.h"

#include "MedImgGLResource/mi_gl_fbo.h"

#include "MedImgRenderAlgorithm/mi_camera_calculator.h"
#include "MedImgRenderAlgorithm/mi_camera_interactor.h"

#include "gl/freeglut.h"


using namespace MED_IMAGING_NAMESPACE;

namespace
{
    std::shared_ptr<OrthoCamera> _camera;
    std::shared_ptr<OrthoCameraInteractor> m_pCameraInteractor;


    int _width = 512;
    int _height = 512;
    int m_iButton = -1;
    Point2 m_ptPre;

    void Init()
    {
        _camera.reset(new OrthoCamera());
        _camera->set_eye(Point3(0,0,700));
        _camera->set_look_at(Point3(0,0,0));
        _camera->set_up_direction(Vector3(0,1,0));
        _camera->set_ortho(-175,175,-175,175,-700,2100);

        m_pCameraInteractor.reset(new OrthoCameraInteractor(_camera));

        
    }

    void Display()
    {
        glViewport(0,0,_width , _height);
        glClearColor(0.0,0.0,0.0,0.0);
        glClear(GL_COLOR_BUFFER_BIT );

        //glEnable(GL_DEPTH_TEST);
        //glDepthFunc(GL_LEQUAL);

        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glPushMatrix();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();


        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        Matrix4 mat_mvp = _camera->get_view_projection_matrix();

        
        glLoadMatrixd(mat_mvp._m);

        glColor3d(1.0,1.0,0.0);


        /*glBegin(GL_QUADS);
        glVertex3d(-50 , -50 , 0);
        glVertex3d(50 , -50 , 0);
        glVertex3d(50 , 50 , 0);
        glVertex3d(-50 , 50 , 0);
        glEnd();*/

        //glutWireSphere(20,20,20);
        //glutSolidCube(20);
        glutSolidCylinder(30 , 100 , 30 ,30);

        glPopAttrib();
        glPopMatrix();

        glutSwapBuffers();
    }

    void Keyboard(unsigned char key , int x , int y)
    {
        switch(key)
        {
        case 't':
            {
                
                break;
            }
        default:
            break;
        }
    }

    void Resize(int x , int y)
    {
        _width = x;
        _height = y;
        m_pCameraInteractor->Resize(_width , _height);
    }

    void Idle()
    {
        glutPostRedisplay();
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

    }

    void MouseMotion(int x , int y)
    {
        Point2 cur_pt(x,y);
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

void UT_MeshRendering(int argc , char* argv[])
{
    try
    {
        glutInit(&argc , argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
        glutInitWindowPosition(0,0);
        glutInitWindowSize(_width,_height);

        glutCreateWindow("Test Mesh Rendering");

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