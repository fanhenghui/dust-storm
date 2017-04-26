#include "mi_graphic_item_voi.h"

#include "MedImgCommon/mi_string_number_converter.h"

#include "MedImgArithmetic/mi_arithmetic_utils.h"

#include "MedImgRenderAlgorithm/mi_mpr_scene.h"
#include "MedImgRenderAlgorithm/mi_volume_infos.h"
#include "MedImgRenderAlgorithm/mi_camera_calculator.h"

#include "mi_model_voi.h"

//Qt
#include <QPen>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneDragDropEvent>

MED_IMAGING_BEGIN_NAMESPACE

GraphicItemVOI::GraphicItemVOI():_pre_item_num(0)
{
}

GraphicItemVOI::~GraphicItemVOI()
{

}

void GraphicItemVOI::set_voi_model(std::shared_ptr<VOIModel> model)
{
    _model = model;
}

std::vector<QGraphicsItem*> GraphicItemVOI::get_init_items()
{
    return std::vector<QGraphicsItem*>();
}

void GraphicItemVOI::update(std::vector<QGraphicsItem*>& to_be_add , std::vector<QGraphicsItem*>& to_be_remove)
{
    try
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(_scene);

        std::shared_ptr<MPRScene> scene = std::dynamic_pointer_cast<MPRScene>(_scene);
        QTWIDGETS_CHECK_NULL_EXCEPTION(scene);

        std::shared_ptr<VolumeInfos> volume_infos = scene->get_volume_infos();
        QTWIDGETS_CHECK_NULL_EXCEPTION(volume_infos);

        int width(1),height(1);
        scene->get_display_size(width , height);

        //////////////////////////////////////////////////////////////////////////
        //Check voi firstly
        const std::list<VOISphere>& voi_list = _model->get_voi_spheres();
        const int voi_count = voi_list.size();

        //graphics item number changed
        if (_pre_item_num != voi_count)
        {
            //Add item
            if (_pre_item_num < voi_count)
            {
                //add to rear
                const int add_num = voi_count - _pre_item_num;
                for(int i = 0; i<add_num ; ++i)
                {
                    GraphicsSphereItem* item = new GraphicsSphereItem();
                    item->set_scene(_scene);
                    item->set_voi_model(_model);
                    item->setPen(QPen(QColor(220,50,50)));
                    item->hide();
                    _items.push_back(item);
                    to_be_add.push_back(item);
                }
            }
            //Delete item
            else
            {
                //delete from head
                int delete_num = _pre_item_num - voi_count;
                for (auto it_del = _items.begin() ; it_del !=  _items.end() ; --delete_num )
                {
                    if (delete_num <= 0)
                    {
                        break;
                    }
                    _items_to_be_delete.push_back(*it_del);
                    to_be_remove.push_back(*it_del);
                    it_del = _items.erase(it_del);
                }
            }
        }
        _pre_item_num = voi_count;

        if (voi_list.empty())
        {
            return;
        }

        //1 Get MPR plane
        std::shared_ptr<CameraBase> camera = scene->get_camera();

        if (_pre_camera == *(std::dynamic_pointer_cast<OrthoCamera>(camera)) && _pre_voi_list == voi_list)
        {
            return;
        }
        else
        {
            _pre_camera = *(std::dynamic_pointer_cast<OrthoCamera>(camera));
            _pre_voi_list = voi_list;
        }


        std::shared_ptr<CameraCalculator> cameraCal = scene->get_camera_calculator();
        Point3 look_at = camera->get_look_at();
        Point3 eye = camera->get_eye();
        Vector3 norm = look_at - eye;
        norm.normalize();
        Vector3 up = camera->get_up_direction();

        const Matrix4 mat_vp = camera->get_view_projection_matrix();
        const Matrix4 mat_p2w = cameraCal->get_patient_to_world_matrix();

        //2 Calculate sphere intersect with plane
        std::vector<Point2> circle_center;
        std::vector<float> radiuses;
        std::vector<int> voi_id;
        Point3 sphere_center;
        double diameter(0.0);

        int idx = 0;
        for (auto it = voi_list.begin() ; it != voi_list.end() ; ++it , ++idx)
        {
            sphere_center = mat_p2w.transform(it->center);
            diameter = it->diameter;
            double distance = norm.dot_product(look_at - sphere_center);
            if (abs(distance) < diameter*0.5)
            {
                Point3 pt0 = sphere_center + distance*norm;
                double radius = sqrt(diameter*diameter*0.25 - distance*distance);
                Point3 pt1 = pt0 + radius*up;
                pt0 = mat_vp.transform(pt0);
                pt1 = mat_vp.transform(pt1);
                Point2 pt_dc0 = ArithmeticUtils::ndc_to_dc_decimal(Point2(pt0.x , pt0.y) , width , height);
                Point2 pt_dc1 = ArithmeticUtils::ndc_to_dc_decimal(Point2(pt1.x , pt1.y) , width , height);
                float radius_float = static_cast<float>( (pt_dc1 - pt_dc0).magnitude() );
                if (radius_float > 0.95f)
                {
                    circle_center.push_back(pt_dc0);
                    radiuses.push_back(radius_float);
                    voi_id.push_back(idx);
                }
            }
        }

        //3 Draw intersect circle if intersected
        auto item_draw = _items.begin();
        for (; item_draw != _items.end() ; ++item_draw)
        {
            if (!(*item_draw)->is_frezze())
            {
                (*item_draw)->hide();
            }
        }
        item_draw = _items.begin();

        for (size_t i = 0 ; i <circle_center.size() ; ++i , ++item_draw)
        {
            if (!(*item_draw)->is_frezze())
            {
                (*item_draw)->set_id(voi_id[i]);
                (*item_draw)->set_sphere(QPointF(static_cast<float>(circle_center[i].x) , static_cast<float>(circle_center[i].y)) , radiuses[i]);
                (*item_draw)->show();
            }
        }
    }
    catch (const Exception& e)
    {
        std::cout << e.what();
        assert(false);
        throw e;
    }
}

void GraphicItemVOI::post_update()
{
    if (_items_to_be_delete.empty())
    {
        return;
    }

    for (auto it = _items_to_be_delete.begin() ; it != _items_to_be_delete.end() ; ++it)
    {
        delete *it;
    }
    std::list<GraphicsSphereItem*>().swap(_items_to_be_delete);
}

MED_IMAGING_END_NAMESPACE

using namespace medical_imaging;

GraphicsSphereItem::GraphicsSphereItem(QGraphicsItem *parent /*= 0 */, QGraphicsScene *scene /*= 0*/):_id(-1),_is_frezze(false)
{
    this->setFlag(QGraphicsItem::ItemIsMovable);
    this->setPen(QPen(QColor(255,0,0)));
}

GraphicsSphereItem::~GraphicsSphereItem()
{
}



void GraphicsSphereItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->button() == Qt::LeftButton)
    {
        QGraphicsEllipseItem::mouseReleaseEvent(event);

        update_circle_center_i();

        frezze(true);
    }
    else if (event->button() == Qt::RightButton)
    {
        update_circle_center_i();

        frezze(true);
    }
}

void GraphicsSphereItem::mouseMoveEvent( QGraphicsSceneMouseEvent *event )
{
    if (event->buttons() == Qt::LeftButton)
    {
        QGraphicsEllipseItem::mouseMoveEvent(event);

        update_circle_center_i();

        update_sphere_center_i();
    }
    else if (event->buttons() == Qt::RightButton)
    {
        float x = abs(this->mapToScene(event->pos()).x() - _pre_center.x());
        float y = abs(_pre_center.y() - this->mapToScene(event->pos()).y());
        float radius = sqrt(x*x + y*y);

        set_sphere(_pre_center , radius);

        update_circle_center_i();

        update_sphere_diameter_i();
    }
}

void GraphicsSphereItem::mouseReleaseEvent( QGraphicsSceneMouseEvent *event )
{
    if (event->button() == Qt::LeftButton)
    {
        QGraphicsEllipseItem::mouseReleaseEvent(event);

        update_sphere_center_i();

        frezze(false);
    }
    else if (event->button() == Qt::RightButton)
    {
        update_sphere_diameter_i();

        frezze(false);
    }
}

void GraphicsSphereItem::dragEnterEvent(QGraphicsSceneDragDropEvent *event)
{
    if (event->buttons() == Qt::LeftButton)
    {
        QGraphicsEllipseItem::dragEnterEvent(event);
    }
    else if (event->buttons() == Qt::RightButton)
    {

    }
}

void GraphicsSphereItem::dragLeaveEvent(QGraphicsSceneDragDropEvent *event)
{
    if (event->buttons() == Qt::LeftButton)
    {
        QGraphicsEllipseItem::dragLeaveEvent(event);
    }
    else if (event->buttons() == Qt::RightButton)
    {

    }
}

void GraphicsSphereItem::dragMoveEvent(QGraphicsSceneDragDropEvent *event)
{
    if (event->buttons() == Qt::LeftButton)
    {

    }
    else if (event->buttons() == Qt::RightButton)
    {

    }
}

void GraphicsSphereItem::update_circle_center_i()
{
    _pre_center = QPointF(this->pos().x() + this->rect().width()*0.5f , this->pos().y() + this->rect().height()*0.5f);
    //std::cout  << "center : " << _pre_center.x()  << " , " << _pre_center.y() << std::endl;
}

void GraphicsSphereItem::set_sphere(QPointF center , float radius)
{
    //////////////////////////////////////////////////////////////////////////
    //Set DC center to ellipse
    this->setRect( QRectF(0, 0, 2*radius ,2*radius));
    this->setPos(center - QPointF(radius,radius));
}

void GraphicsSphereItem::update_sphere_center_i()
{
    //////////////////////////////////////////////////////////////////////////
    //Calculate new sphere center
    QTWIDGETS_CHECK_NULL_EXCEPTION(_scene);

    std::shared_ptr<MPRScene> scene = std::dynamic_pointer_cast<MPRScene>(_scene);
    QTWIDGETS_CHECK_NULL_EXCEPTION(scene);

    std::shared_ptr<CameraBase> camera = scene->get_camera();

    std::shared_ptr<VolumeInfos> volume_infos = scene->get_volume_infos();
    QTWIDGETS_CHECK_NULL_EXCEPTION(volume_infos);

    std::shared_ptr<CameraCalculator> cameraCal = scene->get_camera_calculator();
    Point3 look_at = camera->get_look_at();
    Point3 eye = camera->get_eye();
    Vector3 norm = look_at - eye;
    norm.normalize();
    Vector3 up = camera->get_up_direction();

    const Matrix4 mat_vp = camera->get_view_projection_matrix();
    const Matrix4 mat_vp_inv = mat_vp.get_inverse();
    const Matrix4 mat_p2w = cameraCal->get_patient_to_world_matrix();

    int width(1),height(1);
    scene->get_display_size(width , height);

    //Calculate current circle center world
    Point2 cur_circle_center_ndc = ArithmeticUtils::dc_to_ndc(Point2(_pre_center.x() , _pre_center.y())  , width ,height);
    Point3 cur_circle_center = mat_vp_inv.transform( Point3(cur_circle_center_ndc.x, cur_circle_center_ndc.y , 0.0));

    //Calculate previous circle center world
    VOISphere voi = _model->get_voi_sphere(_id);
    Point3 sphere_center = mat_p2w.transform(voi.center);
    sphere_center = mat_vp.transform(sphere_center);
    sphere_center.z = 0.0;
    Point3 pre_circle_center = mat_vp_inv.transform(sphere_center);

    Vector3 translate = cur_circle_center - pre_circle_center;
    if (translate == Vector3::S_ZERO_VECTOR)
    {
        return;
    }
    else
    {
        Point3 new_center = voi.center + translate;
        _model->modify_voi_sphere_center(_id , new_center);
        _model->notify();
    }
}

void GraphicsSphereItem::update_sphere_diameter_i()
{
    //////////////////////////////////////////////////////////////////////////
    //Calculate new sphere diameter
    QTWIDGETS_CHECK_NULL_EXCEPTION(_scene);

    std::shared_ptr<MPRScene> scene = std::dynamic_pointer_cast<MPRScene>(_scene);
    QTWIDGETS_CHECK_NULL_EXCEPTION(scene);

    std::shared_ptr<CameraBase> camera = scene->get_camera();

    std::shared_ptr<VolumeInfos> volume_infos = scene->get_volume_infos();
    QTWIDGETS_CHECK_NULL_EXCEPTION(volume_infos);

    std::shared_ptr<CameraCalculator> cameraCal = scene->get_camera_calculator();
    Point3 look_at = camera->get_look_at();
    Point3 eye = camera->get_eye();
    Vector3 norm = look_at - eye;
    norm.normalize();
    Vector3 up = camera->get_up_direction();

    const Matrix4 mat_vp = camera->get_view_projection_matrix();
    const Matrix4 mat_vp_inv = mat_vp.get_inverse();
    const Matrix4 mat_p2w = cameraCal->get_patient_to_world_matrix();

    int width(1),height(1);
    scene->get_display_size(width , height);

    //Calculate previous radius
    VOISphere voi = _model->get_voi_sphere(_id);
    const Point3 sphere_center = mat_p2w.transform(voi.center);
    const double diameter = voi.diameter;
    const double distance = norm.dot_product(look_at - sphere_center);
    Point3 pt0 = sphere_center + distance*norm;
    const double radius = sqrt(diameter*diameter*0.25 - distance*distance);
    Point3 pt1 = pt0 + radius*up;
    pt0 = mat_vp.transform(pt0);
    pt1 = mat_vp.transform(pt1);
    const Point2 pt_dc0 = ArithmeticUtils::ndc_to_dc_decimal(Point2(pt0.x , pt0.y) , width , height);
    const Point2 pt_dc1 = ArithmeticUtils::ndc_to_dc_decimal(Point2(pt1.x , pt1.y) , width , height);
    const float pre_radius = static_cast<float>( (pt_dc1 - pt_dc0).magnitude() );

    //Get current circle radius
    const float cur_radius = this->rect().width()*0.5f;

    float radio = abs(cur_radius / pre_radius);
    if (abs(radio - 1.0f) > FLOAT_EPSILON)
    {
        _model->modify_voi_sphere_diameter(_id , voi.diameter*radio);
        _model->notify();
    }

}


void GraphicsSphereItem::set_voi_model(std::shared_ptr<medical_imaging::VOIModel> model)
{
    _model = model;
}

void GraphicsSphereItem::set_scene(std::shared_ptr<medical_imaging::SceneBase> scene)
{
    _scene = scene;
}

void GraphicsSphereItem::frezze(bool flag)
{
    _is_frezze = flag;
}

bool GraphicsSphereItem::is_frezze() const
{
    return _is_frezze;
}

