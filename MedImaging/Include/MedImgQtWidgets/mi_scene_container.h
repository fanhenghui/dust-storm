#ifndef MED_IMAGING_SCENE_CONTAINER_H_
#define MED_IMAGING_SCENE_CONTAINER_H_

#include "MedImgQtWidgets/mi_qt_widgets_stdafx.h"
#include <QtOpenGL/qgl.h>
#include "qt/qpoint.h"
#include "boost/thread/mutex.hpp"

namespace medical_imaging
{
    class SceneBase;
    class PainterBase;
    class IMouseOp;
}

class QPaintEvent;
class QMouseEvent;
class QKeyEvent;

class SharedWidget;

typedef std::shared_ptr<medical_imaging::IMouseOp> IMouseOpPtr;
typedef std::vector<IMouseOpPtr> IMouseOpPtrCollection;
typedef std::shared_ptr<medical_imaging::PainterBase> PainterPtr;

class QtWidgets_Export SceneContainer : public QGLWidget
{
    Q_OBJECT
public:
    SceneContainer(SharedWidget* pShared , QWidget* parent = 0);
    virtual ~SceneContainer();

    void set_name(const std::string& sDes);
    std::string get_name() const;

    void set_scene(std::shared_ptr<medical_imaging::SceneBase> pScene);
    std::shared_ptr<medical_imaging::SceneBase> get_scene();

    void register_mouse_operation(IMouseOpPtr pMouseOP , Qt::MouseButtons eButtons , Qt::KeyboardModifier eKeyBoardModifier);
    void register_mouse_operation(IMouseOpPtrCollection vecMouseOPs , Qt::MouseButtons eButtons , Qt::KeyboardModifier eKeyBoardModifier);
    IMouseOpPtrCollection get_mouse_operation(Qt::MouseButtons eButton , Qt::KeyboardModifier eKeyBoardModifier);

    void register_key_operation();

    void add_painter_list(std::vector<PainterPtr> vecPainters);

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

    void render_to_back_i();

private:
    bool get_mouse_operation_i(QMouseEvent *event , IMouseOpPtrCollection& pOp);

protected:
    std::shared_ptr<medical_imaging::SceneBase> m_pScene;
    std::vector<PainterPtr> m_vecPainters;
    std::map<int , IMouseOpPtrCollection> m_mapMouseOps;
    Qt::MouseButtons m_eButtons;
    std::unique_ptr<QPixmap> m_pPixelMap;
    boost::mutex _mutex;
};

#endif