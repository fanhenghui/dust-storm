#include "mi_graphic_item_voi.h"

#include "MedImgCommon/mi_string_number_converter.h"

#include "MedImgArithmetic/mi_arithmetic_utils.h"

#include "MedImgRenderAlgorithm/mi_mpr_scene.h"
#include "MedImgRenderAlgorithm/mi_volume_infos.h"
#include "MedImgRenderAlgorithm/mi_camera_calculator.h"

#include "mi_model_voi.h"

//Qt
#include <QPen>
#include <QFont>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneDragDropEvent>
#include <QTextDocument>

MED_IMAGING_BEGIN_NAMESPACE

GraphicItemVOI::GraphicItemVOI():_pre_item_num(0),_pre_width(0),_pre_height(0),_item_to_be_tuned(-1)
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
        const std::vector<VOISphere>& vois = _model->get_vois();
        const int voi_count = vois.size();
        //if (this->_model->)
        //{
        //}
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
                    //sphere
                    GraphicsSphereItem* item = new GraphicsSphereItem();
                    item->set_scene(_scene);
                    item->set_voi_model(_model);
                    item->setPen(QPen(QColor(220,50,50)));
                    item->hide();

                    //line
                    GraphicsLineItem* item_line = new GraphicsLineItem();
                    QPen pen(QColor(16,176,51) , 2 , Qt::DotLine);
                    item_line->setPen(pen);

                    //info
                    GraphicsTextItem* item_info = new GraphicsTextItem(item_line);
                    item_info->setDefaultTextColor(QColor(16,176,51));
                    QFont font("Times" , 10 , QFont::Bold);
                    item_info->setFont(font);
                    item_info->setFlags(QGraphicsItem::ItemIsMovable);

                    _items_spheres.push_back(item);
                    _items_infos.push_back(item_info);
                    _items_lines.push_back(item_line);

                    to_be_add.push_back(item);
                    to_be_add.push_back(item_info);
                    to_be_add.push_back(item_line);
                }
            }
            //Delete item
            else
            {
                //delete from rear
                int delete_num = _pre_item_num - voi_count;
                auto it_del_item = (--_items_spheres.end());
                auto it_del_item_info = (--_items_infos.end());
                auto it_del_item_line = (--_items_lines.end());

                while(delete_num >0)
                {
                    _items_to_be_delete.push_back(*it_del_item);
                    to_be_remove.push_back(*it_del_item);
                    it_del_item = _items_spheres.erase(it_del_item);

                    _items_to_be_delete.push_back(*it_del_item_info);
                    to_be_remove.push_back(*it_del_item_info);
                    it_del_item_info = _items_infos.erase(it_del_item_info);

                    _items_to_be_delete.push_back(*it_del_item_line);
                    to_be_remove.push_back(*it_del_item_line);
                    it_del_item_line = _items_lines.erase(it_del_item_line);

                    --delete_num;
                }
            }
        }
        _pre_item_num = voi_count;

        if (vois.empty())
        {
            return;
        }

        const std::vector<IntensityInfo>& intensity_infos = _model->get_intensity_infos();

        //1 Get MPR plane
        std::shared_ptr<CameraBase> camera = scene->get_camera();

        if (_pre_camera == *(std::dynamic_pointer_cast<OrthoCamera>(camera)) &&
            _pre_vois == vois &&
            _pre_intensity_infos == intensity_infos &&
            _pre_width == width &&
            _pre_height == height)
        {
            return;
        }
        else
        {
            _pre_camera = *(std::dynamic_pointer_cast<OrthoCamera>(camera));
            _pre_vois = vois;
            _pre_intensity_infos = intensity_infos;

            if (_pre_width != width || _pre_height != height)
            {
                //Adjust text
                for (int i = 0 ; i<_items_infos.size() ; ++i)
                {
                    _items_infos[i]->grab(false);
                }
                _pre_width = width;
                _pre_height = height;
            }
        }


        std::shared_ptr<CameraCalculator> camera_cal = scene->get_camera_calculator();
        Point3 look_at = camera->get_look_at();
        Point3 eye = camera->get_eye();
        Vector3 norm = look_at - eye;
        norm.normalize();
        Vector3 up = camera->get_up_direction();

        const Matrix4 mat_vp = camera->get_view_projection_matrix();
        const Matrix4 mat_p2w = camera_cal->get_patient_to_world_matrix();

        //2 Calculate sphere intersect with plane
        std::vector<Point2> circle_center;
        std::vector<float> radiuses;
        std::vector<int> voi_id;
        Point3 sphere_center;
        double diameter(0.0);

        int idx = 0;
        for (auto it = vois.begin() ; it != vois.end() ; ++it , ++idx)
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
        for (int i = 0 ; i < _items_spheres.size() ; ++i)
        {
            if (!(_items_spheres[i]->is_frezze()))
            {
                _items_spheres[i]->hide();
            }
            _items_infos[i]->hide();
            _items_lines[i]->hide();
        }

        auto item_sphere = _items_spheres.begin();
        auto item_info = _items_infos.begin();
        auto item_line = _items_lines.begin();

        StrNumConverter<double> str_num_converter;
        for (size_t i = 0 ; i <circle_center.size() ; ++i , ++item_sphere , ++item_info , ++item_line)
        {
            //sphere
            if (!(*item_sphere)->is_frezze())
            {
                (*item_sphere)->set_id(voi_id[i]);
                (*item_sphere)->set_sphere(QPointF(static_cast<float>(circle_center[i].x) , static_cast<float>(circle_center[i].y)) , radiuses[i]);
                (*item_sphere)->show();
            }

            //info
            IntensityInfo info = _model->get_intensity_info(voi_id[i]);
            std::string context = std::string("min : ")  + str_num_converter.to_string_decimal(info._min , 3) +
                std::string(" max : ")  + str_num_converter.to_string_decimal(info._max , 3) + std::string("\n") +
                std::string("mean : ")  + str_num_converter.to_string_decimal(info._mean , 3) + 
                std::string(" std : ")  + str_num_converter.to_string_decimal(info._std, 3) + std::string("\n"); /*+
                std::string("pixel num : ")  + str_num_converter.to_string_decimal(static_cast<double>(info._num),0);*/
            (*item_info)->setPlainText(context.c_str());

            const float sphere_w = (*item_sphere)->rect().width();
            const float sphere_h = (*item_sphere)->rect().height();
            const QPointF sphere_pos = (*item_sphere)->pos();
            QPointF info_pos = (*item_info)->pos();
            if (!(*item_info)->is_grabbed())
            {
                info_pos = sphere_pos +  QPointF(  sphere_w , -sphere_h) + QPointF(30 , -10);
                (*item_info)->setPos(info_pos);
            }
            (*item_info)->show();

            //line
            (*item_line)->setLine(QLineF(sphere_pos + QPointF(sphere_w , 0), info_pos+QPointF(0 , 10.0f*4.0f)));
            (*item_line)->show();
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
    std::vector<QGraphicsItem*>().swap(_items_to_be_delete);
}

void GraphicItemVOI::enable_interaction()
{
    for (int i = 0 ; i < _items_spheres.size() ; ++i)
    {
        this->_items_spheres[i]->setEnabled(true);
        this->_items_infos[i]->setEnabled(true);
        this->_items_lines[i]->setEnabled(true);
    }
}

void GraphicItemVOI::disable_interaction()
{
    for (int i = 0 ; i < _items_spheres.size() ; ++i)
    {
        this->_items_spheres[i]->setEnabled(false);
        this->_items_infos[i]->setEnabled(false);
        this->_items_lines[i]->setEnabled(false);
    }
}

void GraphicItemVOI::set_item_to_be_tuned(const int new_idx)
{
    if(new_idx > static_cast<int>(this->_items_spheres.size()) || new_idx == this->_item_to_be_tuned)
    {
        return;
    }
    if (new_idx >= 0)
    {
        this->_items_spheres[new_idx]->setPen(QPen(QColor(222,203,228)));
    }

    if (this->_item_to_be_tuned >=0)
    {
        this->_items_spheres[this->_item_to_be_tuned]->setPen(QPen(QColor(220,50,50)));
    }

    this->_item_to_be_tuned = new_idx;
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

        update_sphere_center_i(VOIModel::MODIFYING);
    }
    else if (event->buttons() == Qt::RightButton)
    {
        float x = abs(this->mapToScene(event->pos()).x() - _pre_center.x());
        float y = abs(_pre_center.y() - this->mapToScene(event->pos()).y());
        float radius = sqrt(x*x + y*y);

        set_sphere(_pre_center , radius);

        update_circle_center_i();

        update_sphere_diameter_i(VOIModel::MODIFYING);
    }
}

void GraphicsSphereItem::mouseReleaseEvent( QGraphicsSceneMouseEvent *event )
{
    if (event->button() == Qt::LeftButton)
    {
        QGraphicsEllipseItem::mouseReleaseEvent(event);

        update_sphere_center_i(VOIModel::MODIFY_COMPLETED);

        frezze(false);

        
    }
    else if (event->button() == Qt::RightButton)
    {
        update_sphere_diameter_i(VOIModel::MODIFY_COMPLETED);

        frezze(false);

    }
}

std::shared_ptr<medical_imaging::VOIModel>& GraphicsSphereItem::get_voi_model()
{
    return this->_model;
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

void GraphicsSphereItem::update_sphere_center_i(int code_id /*= 0*/)
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
    VOISphere voi = _model->get_voi(_id);
    Point3 sphere_center = mat_p2w.transform(voi.center);
    sphere_center = mat_vp.transform(sphere_center);
    sphere_center.z = 0.0;
    Point3 pre_circle_center = mat_vp_inv.transform(sphere_center);

    Vector3 translate = cur_circle_center - pre_circle_center;
    if (translate == Vector3::S_ZERO_VECTOR && code_id != VOIModel::MODIFY_COMPLETED)
    {
        return;
    }
    else
    {
        Point3 new_center = voi.center + translate;
        _model->modify_center(_id , new_center);
        _model->notify(code_id);
    }
}

void GraphicsSphereItem::update_sphere_diameter_i(int code_id /*= 0*/)
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
    VOISphere voi = _model->get_voi(_id);
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
    const float radio = abs(cur_radius / pre_radius);
    if (abs(radio - 1.0f) < FLOAT_EPSILON && code_id != VOIModel::MODIFY_COMPLETED)
    {
        return;
    }
    else
    {
        _model->modify_diameter(_id , voi.diameter*radio);
        _model->notify(code_id);
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


GraphicsTextItem::GraphicsTextItem(GraphicsLineItem* line_item):_is_grabbed(false)
{
    connect(this , SIGNAL(position_changed(QPointF)) , line_item , SLOT(slot_info_position_changed(QPointF)));
}

GraphicsTextItem::~GraphicsTextItem()
{

}

void GraphicsTextItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->buttons() == Qt::LeftButton)
    {
        _is_grabbed = true;
    }
    QGraphicsTextItem::mousePressEvent(event);
}

void GraphicsTextItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    QGraphicsTextItem::mouseMoveEvent(event);
    emit position_changed(this->pos());
}

GraphicsLineItem::GraphicsLineItem()
{

}

GraphicsLineItem::~GraphicsLineItem()
{

}

void GraphicsLineItem::slot_info_position_changed(QPointF info_pos)
{
    QLineF pre_line = this->line();
    this->setLine(QLineF(pre_line.p1() , info_pos +QPointF(0 , 10.0f*4.0f) ));
}


//////////////////////////////////////////////////////////////////////////
// To simulate the eraser
//////////////////////////////////////////////////////////////////////////
GraphicsCircleItem::GraphicsCircleItem(QGraphicsItem *parent /*= 0 */, QGraphicsScene *scene /*= 0*/)
{

}

GraphicsCircleItem::~GraphicsCircleItem()
{

}

void GraphicsCircleItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    std::shared_ptr<VOIModel>& model = this->get_voi_model();
}

void GraphicsCircleItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{

}

void GraphicsCircleItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{

}