#include "mi_shared_widget.h"


//MED_IMG_BEGIN_NAMESPACE 

boost::mutex SharedWidget::_mutex;

SharedWidget* SharedWidget::_s_instance = nullptr;

SharedWidget* SharedWidget::instance()
{
    if (!_s_instance)
    {
        boost::unique_lock<boost::mutex> locker(_mutex);
        if (!_s_instance)
        {
            _s_instance= new SharedWidget();
            //_s_instance->setWindowFlags(_s_instance->windowFlags() | Qt::FramelessWindowHint);
            //_s_instance->setWindowFlags(Qt::FramelessWindowHint);
            //_s_instance->setAttribute(Qt::WA_TranslucentBackground , true);
            _s_instance->resize(600,370);
            _s_instance->setFixedSize(600,370);
            _s_instance->show();
            _s_instance->hide();
        }
    }
    return _s_instance;
}

SharedWidget::~SharedWidget()
{

}

SharedWidget::SharedWidget(QWidget* parent /*= 0 */, QGLWidget* shared /*= 0*/):QGLWidget(parent , shared)
{

}

void SharedWidget::initializeGL()
{
    try
    {
        if (GLEW_OK != glewInit())
        {
            QTWIDGETS_THROW_EXCEPTION("Glew init failed!");
        }
        glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    }
    catch (const medical_imaging::Exception& e)
    {
        //TODO LOG
        std::cout << e.what();
        assert(false);
        throw e;
    }
}

//MED_IMG_END_NAMESPACE