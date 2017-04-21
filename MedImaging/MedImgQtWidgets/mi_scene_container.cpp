#include "mi_scene_container.h"

#include "qt/qevent.h"
#include "qt/qpainter.h"

#include "MedImgGLResource/mi_gl_utils.h"
#include "MedImgRenderAlgorithm/mi_scene_base.h"

#include "mi_shared_widget.h"
#include "mi_painter_base.h"
#include "mi_mouse_op_interface.h"

using namespace medical_imaging;

SceneContainer::SceneContainer(SharedWidget* pShared , QWidget* parent /*= 0*/):QGLWidget(parent , pShared),m_eButtons(Qt::NoButton),m_pPixelMap(new QPixmap(256,256))
{
    makeCurrent();

    setAutoBufferSwap( false );
    setAutoFillBackground( false );
    //setMouseTracking(true);
}

SceneContainer::~SceneContainer()
{
    makeCurrent();

    /*if (m_pScene)
    {
        m_pScene->finalize();
    }*/
}

void SceneContainer::initializeGL()
{
    if (GLEW_OK != glewInit())
    {
        QTWIDGETS_THROW_EXCEPTION("Glew init failed!");
    }

    if (m_pScene)
    {
        m_pScene->initialize();
    }
}

void SceneContainer::paintEvent(QPaintEvent* pPainter)
{
    boost::unique_lock<boost::mutex> locker(_mutex);
    if (!m_pScene)
    {
        makeCurrent();

        glViewport(0,0,this->width() , this->height());
        glFrontFace( GL_CW );
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glClearColor(0,0,0,0);
        glClear(GL_COLOR_BUFFER_BIT);

        QPainter painter(this);
        painter.setPen(Qt::red);
        painter.drawText(this->width()/2,this->height()/2,tr("NULL"));
        swapBuffers();

        return;
    }
    else
    {
        makeCurrent();

        CHECK_GL_ERROR;
        if (m_pScene)
        {
            m_pScene->initialize();
        }
        CHECK_GL_ERROR;
        //1 Scene rendering
        m_pScene->render(0);
        m_pScene->render_to_back();

        //2 Painter drawing
        m_pPixelMap->fill(Qt::transparent);
        QPainter painter(m_pPixelMap.get());
        for (auto it = m_vecPainters.begin() ; it != m_vecPainters.end()  ;++it)
        {
            (*it)->set_painter(&painter);
            (*it)->render();
        }

        QPainter p(this);
        p.drawPixmap(0,0,*m_pPixelMap);


        //3 Swap buffers
        swapBuffers( );
    }
}

void SceneContainer::mousePressEvent(QMouseEvent *event)
{
    //focus
    this->setFocus();

    m_eButtons = event->buttons();

    IMouseOpPtrCollection vecOps;
    if(get_mouse_operation_i(event , vecOps))
    {
        for (auto it = vecOps.begin() ; it != vecOps.end() ; ++it)
        {
            (*it)->press(event->pos());
        }
    }

    update();
}

void SceneContainer::mouseReleaseEvent(QMouseEvent *event)
{
    IMouseOpPtrCollection vecOps;
    if(get_mouse_operation_i(event , vecOps))
    {
        for (auto it = vecOps.begin() ; it != vecOps.end() ; ++it)
        {
            (*it)->release(event->pos());
        }
    }

    m_eButtons = Qt::NoButton;
    update();
}

void SceneContainer::mouseMoveEvent(QMouseEvent *event)
{
    IMouseOpPtrCollection vecOps;
    if(get_mouse_operation_i(event , vecOps))
    {
        for (auto it = vecOps.begin() ; it != vecOps.end() ; ++it)
        {
            (*it)->move(event->pos());
        }
    }

    update();
}

void SceneContainer::mouseDoubleClickEvent(QMouseEvent *event)
{
    m_eButtons = event->buttons();

    IMouseOpPtrCollection vecOps;
    if(get_mouse_operation_i(event , vecOps))
    {
        for (auto it = vecOps.begin() ; it != vecOps.end() ; ++it)
        {
            (*it)->double_click(event->pos());
        }
    }

    update();
}

void SceneContainer::wheelEvent(QWheelEvent *)
{

}

void SceneContainer::keyPressEvent(QKeyEvent *key)
{
    
}

void SceneContainer::keyReleaseEvent(QKeyEvent *key)
{

}

void SceneContainer::resizeGL(int w, int h)
{
    boost::unique_lock<boost::mutex> locker(_mutex);
    if (m_pScene)
    {
        m_pScene->set_display_size(w,h);
    }
    m_pPixelMap.reset(new QPixmap(w,h));
}

void SceneContainer::set_scene(std::shared_ptr<medical_imaging::SceneBase> pScene)
{
    m_pScene = pScene;
}

void SceneContainer::add_painter_list(std::vector<std::shared_ptr<PainterBase>> vecPainters)
{
    m_vecPainters = vecPainters;
}

void SceneContainer::register_mouse_operation(std::shared_ptr<IMouseOp> pMouseOP , Qt::MouseButtons eButton , Qt::KeyboardModifier eKeyBoardModifier)
{
    m_mapMouseOps[eButton|eKeyBoardModifier] = IMouseOpPtrCollection(1 , pMouseOP);
}

void SceneContainer::register_mouse_operation(IMouseOpPtrCollection vecMouseOPs , Qt::MouseButtons eButton , Qt::KeyboardModifier eKeyBoardModifier)
{
    m_mapMouseOps[eButton|eKeyBoardModifier] = vecMouseOPs;
}

bool SceneContainer::get_mouse_operation_i(QMouseEvent *event , IMouseOpPtrCollection& pOp)
{
    int key = m_eButtons | event->modifiers();
    auto it = m_mapMouseOps.find(key);
    if (it != m_mapMouseOps.end())
    {
        pOp = it->second;
        return true;
    }
    else
    {
        return false;
    }
}

void SceneContainer::set_name(const std::string& sDes)
{
    if (m_pScene)
    {
        m_pScene->set_name(sDes);
    }
}

std::string SceneContainer::get_name() const
{
    if (m_pScene)
    {
        return m_pScene->get_name();
    }
    else
    {
        return "";
    }
}

void SceneContainer::focusInEvent(QFocusEvent *event)
{
    emit focusInScene();
}

void SceneContainer::focusOutEvent(QFocusEvent *event)
{
    emit focusOutScene();
}

IMouseOpPtrCollection SceneContainer::get_mouse_operation(Qt::MouseButtons eButton , Qt::KeyboardModifier eKeyBoardModifier)
{
    int key = eButton | eKeyBoardModifier;
    auto it = m_mapMouseOps.find(key);
    if (it != m_mapMouseOps.end())
    {
        return it->second;
    }
    else
    {
        return IMouseOpPtrCollection();
    }
}
