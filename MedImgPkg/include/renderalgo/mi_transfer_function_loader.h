#ifndef MED_IMG_TRANSFER_LOADER_H
#define MED_IMG_TRANSFER_LOADER_H

#include <memory>
#include <string>
#include "renderalgo/mi_render_algo_export.h"
#include "renderalgo/mi_ray_caster_define.h"
#include "io/mi_io_define.h"

MED_IMG_BEGIN_NAMESPACE 

class ColorTransFunc;
class OpacityTransFunc;

class RenderAlgo_Export TransferFuncLoader
{
public:
    static IOStatus load_pseudo_color(const std::string& xml , std::shared_ptr<ColorTransFunc>& color);

    static IOStatus load_color_opacity(
        const std::string& xml , 
        std::shared_ptr<ColorTransFunc>& color , 
        std::shared_ptr<OpacityTransFunc>& opacity ,
        float &ww , float &wl,
        RGBAUnit& background,
        Material& material );
protected:
private:
};

MED_IMG_END_NAMESPACE

#endif