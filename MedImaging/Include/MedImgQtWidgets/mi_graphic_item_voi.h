#ifndef MED_IMAGING_VOI_PAINTER_H_
#define MED_IMAGING_VOI_PAINTER_H_

#include <list>

#include "MedImgQtWidgets/mi_graphic_item_base.h"
#include "MedImgIO/mi_voi.h"
#include "MedImgArithmetic/mi_ortho_camera.h"
#include <QGraphicsEllipseItem>

namespace medical_imaging
{
    class VOIModel;
    class SceneBase;
}

//Mouse left change position , and right change size
class GraphicsSphereItem : public QGraphicsEllipseItem
{
public:
    GraphicsSphereItem(QGraphicsItem *parent = 0 , QGraphicsScene *scene = 0);
    virtual ~GraphicsSphereItem();

    void set_id(int id) {_id = id;};
    int get_id() const {return _id;};

    void set_sphere(QPointF center , float radius);

    void set_voi_model(std::shared_ptr<medical_imaging::VOIModel> model);

    void set_scene(std::shared_ptr<medical_imaging::SceneBase> scene);

    void frezze(bool flag);
    bool is_frezze() const;

protected:
    virtual void dragEnterEvent(QGraphicsSceneDragDropEvent *event);
    virtual void dragLeaveEvent(QGraphicsSceneDragDropEvent *event);
    virtual void dragMoveEvent(QGraphicsSceneDragDropEvent *event);
    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
private:
    void update_circle_center_i();
    void update_sphere_center_i();
    QPointF _pre_center;
    int _id;
    std::shared_ptr<medical_imaging::VOIModel> _model;
    std::shared_ptr<medical_imaging::SceneBase> _scene;
    bool _is_frezze;
};


MED_IMAGING_BEGIN_NAMESPACE


class QtWidgets_Export GraphicItemVOI : public GraphicItemBase
{
public:
    GraphicItemVOI();
    virtual ~GraphicItemVOI();

    void set_voi_model(std::shared_ptr<VOIModel> model);

    virtual std::vector<QGraphicsItem*> get_init_items();

    virtual void update(std::vector<QGraphicsItem*>& to_be_add , std::vector<QGraphicsItem*>& to_be_remove);

    virtual void post_update();
private:

private:
    std::shared_ptr<VOIModel> _model;
    std::list<GraphicsSphereItem*> _items;
    int _pre_item_num;
    std::list<GraphicsSphereItem*> _items_to_be_delete;

    std::list<VOISphere> _pre_voi_list;
    OrthoCamera _pre_camera;
};

MED_IMAGING_END_NAMESPACE

#endif