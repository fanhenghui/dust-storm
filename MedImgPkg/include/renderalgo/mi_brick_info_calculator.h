#ifndef MED_IMG_BRICK_INFO_CALCULATOR
#define MED_IMG_BRICK_INFO_CALCULATOR

#include <memory>
#include "renderalgo/mi_render_algo_export.h"
#include "renderalgo/mi_brick_define.h"
#include "glresource/mi_gl_resource_define.h"
#include "glresource/mi_gl_resource_manager_container.h"
#include "arithmetic/mi_aabb.h"

MED_IMG_BEGIN_NAMESPACE 

class ImageData;
class BrickPool;

class RenderAlgo_Export BrickInfoCalculator
{
public:
    BrickInfoCalculator();
    virtual ~BrickInfoCalculator();

    void set_data(std::shared_ptr<ImageData> img);
    void set_data_texture(GLTexture3DPtr tex);
    void set_brick_info_buffer(GLBufferPtr info_buffer);
    void set_brick_info_array(char* info_array);

    void set_brick_size(unsigned int brick_size);
    void set_brick_dim(unsigned int (&brick_dim)[3]);
    void set_brick_margin(unsigned int brick_margin);

    virtual void calculate() = 0;

protected:
    std::shared_ptr<ImageData> _img_data;
    GLTexture3DPtr _img_texture;
    GLBufferPtr _info_buffer;
    char* _info_array;

    unsigned int _brick_size;
    unsigned int _brick_dim[3];
    unsigned int _brick_margin;

    GLResourceShield _res_shield;

private:
    DISALLOW_COPY_AND_ASSIGN(BrickInfoCalculator);
};

class RenderAlgo_Export VolumeBrickInfoCalculator : public BrickInfoCalculator
{
public:
    VolumeBrickInfoCalculator();
    virtual ~VolumeBrickInfoCalculator();

    virtual void calculate();

private:
    void initialize_i();
    void calculate_i();
    void download_i();

private:
    GLProgramPtr _gl_program;

};

class RenderAlgo_Export MaskBrickInfoCalculator : public BrickInfoCalculator
{
public:
    MaskBrickInfoCalculator();
    virtual ~MaskBrickInfoCalculator();

    void set_visible_labels(const std::vector<unsigned char>& labels);

    virtual void calculate();
    void update(const AABBUI& aabb);

private:
    void initialize_i();
    void calculate_i();
    void update_i(const AABBUI& aabb);
    void download_i();

private:
    GLProgramPtr _gl_program;
    GLBufferPtr _gl_buffer_visible_labels;
    std::vector<unsigned char> _visible_labels;
};

MED_IMG_END_NAMESPACE


#endif