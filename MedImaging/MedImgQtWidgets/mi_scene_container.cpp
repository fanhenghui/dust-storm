#include "mi_scene_container.h"

#include "qt/qevent.h"
#include "qt/qpainter.h"

#include "MedImgGLResource/mi_gl_utils.h"
#include "MedImgRenderAlgorithm/mi_scene_base.h"

#include "mi_shared_widget.h"
#include "mi_painter_base.h"
#include "mi_mouse_op_interface.h"

using namespace MedImaging;

SceneContainer::SceneContainer(SharedWidget* pShared , QWidget* parent /*= 0*/):QGLWidget(parent , pShared),m_eButton(Qt::NoButton),m_pPixelMap(new QPixmap(256,256))
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
        m_pScene->Finalize();
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
        m_pScene->Initialize();
    }
}

void SceneContainer::paintEvent(QPaintEvent* pPainter)
{
    boost::unique_lock<boost::mutex> locker(m_mutex);
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
            m_pScene->Initialize();
        }
        CHECK_GL_ERROR;
        //1 Scene rendering
        m_pScene->Render(0);
        m_pScene->RenderToBack();

        //2 Painter drawing
        m_pPixelMap->fill(Qt::transparent);
        QPainter painter(m_pPixelMap.get());
        for (auto it = m_vecPainters.begin() ; it != m_vecPainters.end()  ;++it)
        {
            (*it)->SetQPainter(&painter);
            (*it)->Render();
        }

        QPainter p(this);
        p.drawPixmap(0,0,*m_pPixelMap);


        //3 Swap buffers
        swapBuffers( );
    }
}

void SceneContainer::mousePressEvent(QMouseEvent *event)
{
    //Focus
    this->setFocus();

    m_eButton = event->button();

    IMouseOpPtrCollection vecOps;
    if(GetMouseOperation_i(event , vecOps))
    {
        for (auto it = vecOps.begin() ; it != vecOps.end() ; ++it)
        {
            (*it)->Press(event->pos());
        }
    }

    update();
}

void SceneContainer::mouseReleaseEvent(QMouseEvent *event)
{
    IMouseOpPtrCollection vecOps;
    if(GetMouseOperation_i(event , vecOps))
    {
        for (auto it = vecOps.begin() ; it != vecOps.end() ; ++it)
        {
            (*it)->Release(event->pos());
        }
    }

    m_eButton = Qt::NoButton;
    update();
}

void SceneContainer::mouseMoveEvent(QMouseEvent *event)
{
    IMouseOpPtrCollection vecOps;
    if(GetMouseOperation_i(event , vecOps))
    {
        for (auto it = vecOps.begin() ; it != vecOps.end() ; ++it)
        {
            (*it)->Move(event->pos());
        }
    }

    update();
}

void SceneContainer::mouseDoubleClickEvent(QMouseEvent *event)
{
    m_eButton = event->button();

    IMouseOpPtrCollection vecOps;
    if(GetMouseOperation_i(event , vecOps))
    {
        for (auto it = vecOps.begin() ; it != vecOps.end() ; ++it)
        {
            (*it)->DoubleClick(event->pos());
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
    boost::unique_lock<boost::mutex> locker(m_mutex);
    if (m_pScene)
    {
        m_pScene->SetDisplaySize(w,h);
    }
    m_pPixelMap.reset(new QPixmap(w,h));
}

void SceneContainer::SetScene(std::shared_ptr<MedImaging::SceneBase> pScene)
{
    m_pScene = pScene;
}

void SceneContainer::AddPainterList(std::vector<std::shared_ptr<PainterBase>> vecPainters)
{
    m_vecPainters = vecPainters;
}

void SceneContainer::RegisterMouseOperation(std::shared_ptr<IMouseOp> pMouseOP , Qt::MouseButton eButton , Qt::KeyboardModifier eKeyBoardModifier)
{
    m_mapMouseOps[eButton|eKeyBoardModifier] = IMouseOpPtrCollection(1 , pMouseOP);
}

void SceneContainer::RegisterMouseOperation(IMouseOpPtrCollection vecMouseOPs , Qt::MouseButton eButton , Qt::KeyboardModifier eKeyBoardModifier)
{
    m_mapMouseOps[eButton|eKeyBoardModifier] = vecMouseOPs;
}

bool SceneContainer::GetMouseOperation_i(QMouseEvent *event , IMouseOpPtrCollection& pOp)
{
    int key = m_eButton | event->modifiers();
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

void SceneContainer::SetName(const std::string& sDes)
{
    if (m_pScene)
    {
        m_pScene->SetName(sDes);
    }
}

std::string SceneContainer::GetName() const
{
    if (m_pScene)
    {
        return m_pScene->GetName();
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

IMouseOpPtrCollection SceneContainer::GetMouseOperation(Qt::MouseButton eButton , Qt::KeyboardModifier eKeyBoardModifier)
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
