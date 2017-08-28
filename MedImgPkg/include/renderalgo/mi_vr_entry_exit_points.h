#ifndef MED_IMG_RENDER_VR_ENTRY_EXIT_POINTS_H
#define MED_IMG_RENDER_VR_ENTRY_EXIT_POINTS_H

#include "renderalgo/mi_entry_exit_points.h"
#include "arithmetic/mi_aabb.h"
#include "arithmetic/mi_plane.h"
#include "arithmetic/mi_vector2f.h"
#include "glresource/mi_gl_resource_define.h"
#include "renderalgo/mi_ray_caster_define.h"

MED_IMG_BEGIN_NAMESPACE 

enum BrickFilterItem
{
    BF_MASK = 1,
    BF_WL = 2,//If has SSD should fix confilict with wl(item pos repeated)
};

enum BrickCullItem
{
    BC_DEPTH = 1,//other graphic object render before ray cast
    BC_PLANE = 2,//clipping plane
};

class BrickGeometry;
class BrickPool;
class ProxyGeometryCube;
class ProxyGeometryBrick;
class RenderAlgo_Export VREntryExitPoints : public EntryExitPoints
{
    friend class ProxyGeometryCube;
    friend class ProxyGeometryBrick;
public:
    VREntryExitPoints();
    virtual ~VREntryExitPoints();

    virtual void initialize();

    virtual void set_display_size(int width , int height);

    virtual void calculate_entry_exit_points();

    void set_brick_pool(std::shared_ptr<BrickPool> brick_pool);

    void set_proxy_geometry(ProxyGeometry pg);
    void set_brick_filter_item(int items);//BrickFilterItem |
    void set_brick_cull_item(int items);//BrickCullItem |

    void set_bounding_box(const AABB& aabb);
    void set_clipping_plane(std::vector<Plane> planes);
    void set_visible_mask(std::vector<unsigned char>& labels);

    void set_window_level(float ww , float wl , unsigned char label , bool global = false);

private:
    std::shared_ptr<BrickPool> _brick_pool;

    ProxyGeometry _proxy_geometry;

    int _brick_filter_items;
    int _brick_cull_items;

    AABB _aabb;
    std::vector<unsigned char> _vis_labels;

    GLFBOPtr _gl_fbo;
    GLTexture2DPtr _gl_depth_texture;

    std::shared_ptr<ProxyGeometryCube> _proxy_geo_cube;
    std::shared_ptr<ProxyGeometryBrick> _proxy_geo_brick;

    std::map<unsigned char , Vector2f> _window_levels;



private:
    DISALLOW_COPY_AND_ASSIGN(VREntryExitPoints);
};

MED_IMG_END_NAMESPACE


#endif