#ifndef MED_IMAGING_SHARED_WIDGET_H_
#define MED_IMAGING_SHARED_WIDGET_H_


#include "MedImgQtWidgets/mi_qt_widgets_stdafx.h"
#include "gl/glew.h"
#include "QtOpenGL/qgl.h"
#include "boost/thread/mutex.hpp"

//MED_IMAGING_BEGIN_NAMESPACE

class QtWidgets_Export SharedWidget : public QGLWidget
{
    Q_OBJECT
public:
    static SharedWidget* instance();
    ~SharedWidget();
protected:
    virtual void initializeGL();
private:
    SharedWidget(QWidget* parent = 0  , QGLWidget* shared = 0);
    static SharedWidget* _s_instance;
    static boost::mutex _mutex;

};
//MED_IMAGING_END_NAMESPACE

#endif