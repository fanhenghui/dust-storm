#include "gl/glew.h"

#include "MedImgArithmetic/mi_ortho_camera.h"

#include "MedImgGLResource/mi_gl_fbo.h"

#include "MedImgRenderAlgorithm/mi_camera_calculator.h"
#include "MedImgRenderAlgorithm/mi_camera_interactor.h"

#include "gl/freeglut.h"


using namespace MED_IMAGING_NAMESPACE;

namespace
{
    std::shared_ptr<OrthoCamera> m_pCamera;
    std::shared_ptr<OrthoCameraInteractor> m_pCameraInteractor;


    int m_iWidth = 512;
    int m_iHeight = 512;
    int m_iButton = -1;
    Point2 m_ptPre;

    void Init()
    {
        m_pCamera.reset(new OrthoCamera());
        m_pCamera->set_eye(Point3(0,0,700));
        m_pCamera->set_look_at(Point3(0,0,0));
        m_pCamera->set_up_direction(Vector3(0,1,0));
        m_pCamera->set_ortho(-175,175,-175,175,-700,2100);

        m_pCameraInteractor.reset(new OrthoCameraInteractor(m_pCamera));

        
    }

    void Display()
    {
        glViewport(0,0,m_iWidth , m_iHeight);
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

        Matrix4 matMVP = m_pCamera->get_view_projection_matrix();

        
        glLoadMatrixd(matMVP._m);

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
        m_iWidth = x;
        m_iHeight = y;
        m_pCameraInteractor->Resize(m_iWidth , m_iHeight);
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
        Point2 ptCur(x,y);
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

void UT_MeshRendering(int argc , char* argv[])
{
    try
    {
        glutInit(&argc , argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
        glutInitWindowPosition(0,0);
        glutInitWindowSize(m_iWidth,m_iHeight);

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