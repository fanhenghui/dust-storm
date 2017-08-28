#ifndef MED_IMG_RAY_CASTING_CPU_H_
#define MED_IMG_RAY_CASTING_CPU_H_

#include <memory>
#include "renderalgo/mi_render_algo_export.h"
#include "arithmetic/mi_color_unit.h"
#include "arithmetic/mi_vector4f.h"
#include "arithmetic/mi_sampler.h"

MED_IMG_BEGIN_NAMESPACE

class RayCaster;
//Just MPR 
class RayCastingCPU
{
public:
    RayCastingCPU(std::shared_ptr<RayCaster> ray_caster);
    ~RayCastingCPU();

    void render(int test_code = 0);

private:
    //For testing
    void render_entry_exit_points_i(int test_code);

    //Dispatch render mode
    template<class T>
    void ray_casting_i(std::shared_ptr<RayCaster> ray_caster);

    //Average
    template<class T>
    void ray_casting_average_i(std::shared_ptr<RayCaster> ray_caster);

    //MIP
    template<class T>
    void ray_casting_mip_i(std::shared_ptr<RayCaster> ray_caster);

    //MinIP
    template<class T>
    void ray_casting_minip_i(std::shared_ptr<RayCaster> ray_caster);

    void mask_overlay_i(std::shared_ptr<RayCaster> ray_caster);

private:
    std::weak_ptr<RayCaster> _ray_caster;
    //Cache
    int _width;
    int _height;
    Vector4f* _entry_points;
    Vector4f* _exit_points;
    unsigned int _dim[3];
    void* _volume_data_array;
    unsigned char* _mask_data_array;
    RGBAUnit* _canvas_array;

};



MED_IMG_END_NAMESPACE

#endif