#include "gl/glew.h"
#include <iostream>
#include "my_gl_widget.h"

#include <QPainter>

MyGLWidget::MyGLWidget(QWidget *parent ):QGLWidget(parent)
{
    makeCurrent();

    setAutoBufferSwap( false );
    setAutoFillBackground( false );

    resize(200 , 200);

}

MyGLWidget::~MyGLWidget()
{
    makeCurrent();
}

void MyGLWidget::initializeGL()
{
    GLenum err = glewInit();
    if (err != GLEW_OK)
    {
        std::cout << "glew init failed!\n";
    }
}

void MyGLWidget::paintEvent( QPaintEvent* pPainter )
{
    makeCurrent();
    glFrontFace( GL_CW );
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glClearColor(0,0,0,0);
    glClear(GL_COLOR_BUFFER_BIT);

    glBegin(GL_TRIANGLES);
    glColor3d(1.0,0.0,0.0);
    glVertex2f(-0.5 ,-0.5);
    glColor3d(0.0,1.0,0.0);
    glVertex2f(0.5 ,-0.5);
    glColor3d(0.0,0.0,1.0);
    glVertex2f(0.5 ,0.5);
    glEnd();

    glPopAttrib();

    QPainter painter(this);
    painter.setPen(Qt::gray);
    painter.drawText(20, 20, "Test Patient name");
    swapBuffers( );
}

void MyGLWidget::resizeGL(int w, int h)
{

}



