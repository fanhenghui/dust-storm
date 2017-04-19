#ifndef MY_GL_WIDGET_H_
#define MY_GL_WIDGET_H_

#include "qt/qgl.h"

class MyGLWidget : public QGLWidget
{
    Q_OBJECT
public:
    MyGLWidget(QWidget *parent = 0);
    ~MyGLWidget();
protected:
    void paintEvent(QPaintEvent* pPainter);
    virtual void resizeGL(int w, int h);
    virtual void initializeGL();
private:
};

#endif