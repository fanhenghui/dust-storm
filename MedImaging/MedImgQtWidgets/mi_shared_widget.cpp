#include "mi_shared_widget.h"


//MED_IMAGING_BEGIN_NAMESPACE

boost::mutex SharedWidget::m_mutex;

SharedWidget* SharedWidget::m_instance = nullptr;

SharedWidget* SharedWidget::Instance()
{
    if (!m_instance)
    {
        boost::unique_lock<boost::mutex> locker(m_mutex);
        if (!m_instance)
        {
            m_instance= new SharedWidget();
            //m_instance->setWindowFlags(m_instance->windowFlags() | Qt::FramelessWindowHint);
            m_instance->resize(600,370);
            m_instance->setFixedSize(600,370);
            m_instance->show();
            m_instance->hide();
        }
    }
    return m_instance;
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
    }
    catch (const MedImaging::Exception& e)
    {
        //TODO LOG
        std::cout << e.what();
        assert(false);
        throw e;
    }
}

//MED_IMAGING_END_NAMESPACE