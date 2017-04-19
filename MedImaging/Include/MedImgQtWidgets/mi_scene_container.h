#ifndef MED_IMAGING_SCENE_CONTAINER_H_
#define MED_IMAGING_SCENE_CONTAINER_H_

#include "MedImgQtWidgets/mi_qt_widgets_stdafx.h"
#include <QtOpenGL/qgl.h>
#include "qt/qpoint.h"
#include "boost/thread/mutex.hpp"

namespace MedImaging
{
    class SceneBase;
    class PainterBase;
    class IMouseOp;
}

class QPaintEvent;
class QMouseEvent;
class QKeyEvent;

class SharedWidget;

typedef std::shared_ptr<MedImaging::IMouseOp> IMouseOpPtr;
typedef std::vector<IMouseOpPtr> IMouseOpPtrCollection;
typedef std::shared_ptr<MedImaging::PainterBase> PainterPtr;

class QtWidgets_Export SceneContainer : public QGLWidget
{
    Q_OBJECT
public:
    SceneContainer(SharedWidget* pShared , QWidget* parent = 0);
    virtual ~SceneContainer();
    void SetName(const std::string& sDes);
    std::string GetName() const;
    void SetScene(std::shared_ptr<MedImaging::SceneBase> pScene);
    std::shared_ptr<MedImaging::SceneBase> GetScene();
    void RegisterMouseOperation(IMouseOpPtr pMouseOP , Qt::MouseButton eButton , Qt::KeyboardModifier eKeyBoardModifier);
    void RegisterMouseOperation(IMouseOpPtrCollection vecMouseOPs , Qt::MouseButton eButton , Qt::KeyboardModifier eKeyBoardModifier);
    IMouseOpPtrCollection GetMouseOperation(Qt::MouseButton eButton , Qt::KeyboardModifier eKeyBoardModifier);
    void RegisterKeyOperation();
    void AddPainterList(std::vector<PainterPtr> vecPainters);

signals:
    void focusInScene();
    void focusOutScene();

protected:
    //////////////////////////////////////////////////////////////////////////
    //Qt virtual function for interaction 
    virtual void initializeGL();
    virtual void resizeGL(int w, int h);
    virtual void paintEvent(QPaintEvent* pPainter);
    virtual void mousePressEvent(QMouseEvent *event);
    virtual void mouseReleaseEvent( QMouseEvent *event );
    virtual void mouseMoveEvent(QMouseEvent *event);
    virtual void mouseDoubleClickEvent(QMouseEvent *event);
    virtual void wheelEvent(QWheelEvent *);
    virtual void keyPressEvent(QKeyEvent *key);
    virtual void keyReleaseEvent(QKeyEvent *key);
    virtual void focusInEvent(QFocusEvent *event);
    virtual void focusOutEvent(QFocusEvent *event);
    ////////////////////////////////////////////////////////////////////////// 

    void RenderToBack_i();

private:
    bool GetMouseOperation_i(QMouseEvent *event , IMouseOpPtrCollection& pOp);

protected:
    std::shared_ptr<MedImaging::SceneBase> m_pScene;
    std::vector<PainterPtr> m_vecPainters;
    std::map<int , IMouseOpPtrCollection> m_mapMouseOps;
    Qt::MouseButton m_eButton;
    std::unique_ptr<QPixmap> m_pPixelMap;
    boost::mutex m_mutex;
};

#endif