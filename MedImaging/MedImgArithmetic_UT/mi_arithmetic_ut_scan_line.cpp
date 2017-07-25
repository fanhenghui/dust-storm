#include "gl/glew.h"
#include "gl/glut.h"
#include "gl/GLU.h"

#include <fstream>

#include "MedImgArithmetic/mi_scan_line_analysis.h"
#include "MedImgArithmetic/mi_color_unit.h"

using namespace medical_imaging;

namespace
{
    int _width = 512;
    int _height = 512;

     RGBUnit* _img_mask = nullptr;
     ScanLineAnalysis<RGBUnit> _scan_line_anaysis;
     std::vector<std::vector<ScanLineAnalysis<RGBUnit>::Pt2> > _pre_contours;
     std::vector<ScanLineAnalysis<RGBUnit>::Pt2> _cur_contours;


    void display()
    {
        glViewport(0,0,_width,_height);
        glClearColor(0,0,0,0);
        glClear(GL_COLOR_BUFFER_BIT );

        glDrawPixels(_width , _height , GL_RGB , GL_UNSIGNED_BYTE , _img_mask);

        glPushMatrix();

        /*glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(0,0,100,0,0,0,0,1,0);*/
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluOrtho2D(0,_width ,0,_height);

        glPointSize(2.0);
        //if (!_pre_contours.empty())
        //{
        //    glColor3f(0.0,1.0,1.0);
        //    glBegin(GL_POINTS);
        //    for (int i = 0 ; i<_pre_contours.size() ; ++i)
        //    {
        //        for (int j = 0 ; j < _pre_contours[i].size() ; ++j)
        //        {
        //            glVertex2d(_pre_contours[i][j].x , _pre_contours[i][j].y);
        //        }
        //    }
        //    glEnd();
        //}

        if (!_cur_contours.empty())
        {
            glColor3f(1.0,1.0,0.0);
            glBegin(GL_POINTS);
            for (int j = 0 ; j < _cur_contours.size() ; ++j)
            {
                glVertex2d(_cur_contours[j].x , _cur_contours[j].y);
            }
            glEnd();

        }

         glPopMatrix();

        glutSwapBuffers();
    }

    void reshape(int x , int y)
    {
        if (x == 0 || y == 0)
        {
            return;
        }
        _width = x;
        _height = y;
        
        if (nullptr != _img_mask)
        {
            delete [] _img_mask;
        }
        _img_mask = new RGBUnit[x*y];
        memset((char*)_img_mask , 0 , x*y*sizeof(RGBUnit));

        //RGBUnit red;
        //red.r = 255;
        //for (int y = 0; y< _height/2 ; ++y)
        //{
        //    for (int x = 0 ; x < _width/2 ; ++x)
        //    {
        //        _img_mask[y*_width + x ] = red;
        //    }
        //}


        glutPostRedisplay();
    }

    void motion(int x , int y)
    {
        ScanLineAnalysis<RGBUnit>::Pt2 pt;
        pt.x = x;
        pt.y = _height - y;
        _cur_contours.push_back(pt);
        glutPostRedisplay();

    }

    void mouse(int btn , int status , int x , int y)
    {
        if (status == GLUT_DOWN)
        {
            
        }
        else if(status == GLUT_UP)
        {
            if (!_cur_contours.empty())
            {
                //scan line 
                RGBUnit red;
                red.r = 255;
                _scan_line_anaysis.fill(_img_mask , _width , _height , _cur_contours , red);
                _pre_contours.push_back(_cur_contours);
                _cur_contours.clear();
            }
        }
        glutPostRedisplay();

    }
}

void UT_ScanLine(int argc , char* argv[])
{
    glutInit(&argc , argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(0,0);
    glutInitWindowSize(_width,_height);

    glutCreateWindow("Test Scan Line");

    if ( GLEW_OK != glewInit())
    {
        std::cout <<"Init glew failed!\n";
        return;
    }

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);

    glutMainLoop(); 

}