#ifndef MED_IMAGING_VOI_PAINTER_H_
#define MED_IMAGING_VOI_PAINTER_H_

#include <vector>

#include "qtpackage/mi_graphic_item_base.h"
#include "io/mi_voi.h"
#include "arithmetic/mi_ortho_camera.h"
#include "arithmetic/mi_volume_statistician.h"

#include <QGraphicsEllipseItem>
#include <QGraphicsTextItem>
#include <QGraphicsLineItem>
#include <QObject>


namespace medical_imaging
{
    class VOIModel;
    class SceneBase;
}


class GraphicsLineItem : public QObject , public QGraphicsLineItem
{
    Q_OBJECT
public:
    GraphicsLineItem();
    virtual ~GraphicsLineItem();

protected slots:
    void slot_info_position_changed(QPointF info_pos);

private:
};

class GraphicsTextItem : public QGraphicsTextItem
{
    Q_OBJECT
public:
    GraphicsTextItem(GraphicsLineItem* line_item);
    virtual ~GraphicsTextItem();

    void set_id(int id) {_id = id;};
    int get_id() const {return _id;};

    bool is_grabbed() {return _is_grabbed;};
    void grab(bool flag) {_is_grabbed = flag;};

signals:
    void position_changed(QPointF info_pos);

protected:
    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
private:
    int _id;
    bool _is_grabbed;
};


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
    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
    std::shared_ptr<medical_imaging::VOIModel>& get_voi_model();

private:
    void update_circle_center_i();
    void update_sphere_center_i(int code_id = 0);
    void update_sphere_diameter_i(int code_id = 0);
    QPointF _pre_center;
    int _id;
    std::shared_ptr<medical_imaging::VOIModel> _model;
    std::shared_ptr<medical_imaging::SceneBase> _scene;
    bool _is_frezze;
};

// left-button changes circle size
class GraphicsCircleItem : public GraphicsSphereItem
{
public:
    GraphicsCircleItem(QGraphicsItem *parent = 0 , QGraphicsScene *scene = 0);
    virtual ~GraphicsCircleItem();

protected:
    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
};


MED_IMG_BEGIN_NAMESPACE

class QtWidgets_Export GraphicItemVOI : public GraphicItemBase
{
public:
    GraphicItemVOI();

    virtual ~GraphicItemVOI();

    void set_voi_model(std::shared_ptr<VOIModel> model);

    virtual std::vector<QGraphicsItem*> get_init_items();

    virtual void update(std::vector<QGraphicsItem*>& to_be_add , std::vector<QGraphicsItem*>& to_be_remove);

    virtual void post_update();

    void enable_interaction();

    void disable_interaction();

    void set_item_to_be_tuned(const int idx);

private:
    std::shared_ptr<VOIModel> _model;
    std::vector<GraphicsSphereItem*> _items_spheres;//Circle on MPR plane
    std::vector<GraphicsTextItem*> _items_infos;//Intensity info
    std::vector<GraphicsLineItem*> _items_lines;//Circle to info line
    int _pre_item_num;
    std::vector<QGraphicsItem*> _items_to_be_delete;

    // three circles on three slices
    std::unique_ptr<GraphicsSphereItem> _tune_graphic_item;
    int _item_to_be_tuned;

    //cache
    std::vector<VOISphere> _pre_vois;
    std::vector<IntensityInfo> _pre_intensity_infos;
    OrthoCamera _pre_camera;
    int _pre_width;
    int _pre_height;
};

MED_IMG_END_NAMESPACE

#endif