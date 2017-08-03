#include "GL/glew.h"
#include "mi_brick_info_calculator.h"
#include "MedImgGLResource/mi_gl_texture_3d.h"
#include "MedImgGLResource/mi_gl_buffer.h"
#include "MedImgGLResource/mi_gl_program.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "MedImgIO/mi_image_data.h"

#include "mi_shader_collection.h"

MED_IMG_BEGIN_NAMESPACE

BrickInfoCalculator::BrickInfoCalculator():_info_array(nullptr),_brick_size(16),_brick_margin(2)
{
    _brick_dim[0] = 1;
    _brick_dim[1] = 1;
    _brick_dim[2] = 1;
}

BrickInfoCalculator::~BrickInfoCalculator()
{

}

void BrickInfoCalculator::set_brick_margin(unsigned int brick_margin)
{
    _brick_margin = brick_margin;
}

void BrickInfoCalculator::set_brick_dim(unsigned int (&brick_dim)[3])
{
    memcpy(_brick_dim , brick_dim , sizeof(unsigned int)*3);
}

void BrickInfoCalculator::set_brick_size(unsigned int brick_size)
{
    _brick_size = brick_size;
}

void BrickInfoCalculator::set_brick_info_array(char* info_array)
{
    _info_array = info_array;
}

void BrickInfoCalculator::set_brick_info_buffer(GLBufferPtr info_buffer)
{
    _info_buffer = info_buffer;
}

void BrickInfoCalculator::set_data_texture(GLTexture3DPtr tex)
{
    _img_texture = tex;
}

void BrickInfoCalculator::set_data(std::shared_ptr<ImageData> img)
{
    _img_data = img;
}

VolumeBrickInfoCalculator::VolumeBrickInfoCalculator()
{

}

VolumeBrickInfoCalculator::~VolumeBrickInfoCalculator()
{

}

void VolumeBrickInfoCalculator::calculate()
{
    initialize_i();
    calculate_i();
    download_i();
}

void VolumeBrickInfoCalculator::download_i()
{
    _info_buffer->set_buffer_target(GL_SHADER_STORAGE_BUFFER);
    _info_buffer->bind();
    _info_buffer->download(_brick_dim[0]*_brick_dim[1]*_brick_dim[2]*sizeof(VolumeBrickInfo) , _info_array);
    _info_buffer->unbind();
}

void VolumeBrickInfoCalculator::calculate_i()
{
    RENDERALGO_CHECK_NULL_EXCEPTION(_img_data);
    RENDERALGO_CHECK_NULL_EXCEPTION(_img_texture);
    RENDERALGO_CHECK_NULL_EXCEPTION(_info_buffer);
    RENDERALGO_CHECK_NULL_EXCEPTION(_info_array);

#define BRICK_VOLUME_INFO_BUFFER 0
#define BRICK_SIZE 1
#define BRICK_MARGIN 2
#define BRICK_DIM 3
#define VOLUME_DIM 4
#define VOLUME_TEXTURE 5
#define VOLUME_MIN_SCALAR 6
#define VOLUME_REGULATE_PARAMETER 7

    CHECK_GL_ERROR;

    _gl_program->bind();

    const unsigned int program_id = _gl_program->get_id();

    glProgramUniform1i(program_id, BRICK_SIZE , _brick_size);
    glProgramUniform1i(program_id , BRICK_MARGIN , _brick_margin);
    glProgramUniform3i(program_id , BRICK_DIM , _brick_dim[0] , _brick_dim[1] , _brick_dim[2]);
    glProgramUniform3i(program_id , VOLUME_DIM ,static_cast<GLint>(_img_data->_dim[0]) , static_cast<GLint>(_img_data->_dim[1]) , static_cast<GLint>(_img_data->_dim[2]));
    glProgramUniform1f(program_id , VOLUME_MIN_SCALAR , _img_data->get_min_scalar());

    float volume_reg_param = 65535.0f;
    if (USHORT == _img_data->_data_type ||
        SHORT == _img_data->_data_type)
    {
        volume_reg_param = 65535.0f;
    }
    else if (UCHAR == _img_data->_data_type ||
        CHAR == _img_data->_data_type)
    {
        volume_reg_param = 256.0f;
    }
    else
    {
        volume_reg_param = 1.0f;
    }
    glProgramUniform1f(program_id , VOLUME_REGULATE_PARAMETER , volume_reg_param);

    _img_texture->bind();
    glActiveTexture(GL_TEXTURE0);
    GLTextureUtils::set_3d_wrap_s_t_r(GL_CLAMP_TO_BORDER);
    GLTextureUtils::set_filter(GL_TEXTURE_3D , GL_LINEAR);
    glProgramUniform1i(program_id , VOLUME_TEXTURE , 0);

    _info_buffer->bind_buffer_base(GL_SHADER_STORAGE_BUFFER , BRICK_VOLUME_INFO_BUFFER);

    const unsigned int local_group_size[3] = {5,5,1};//This parameter is optimal in ***
    unsigned int global_group_size[3] = {0};
    for (int i = 0; i< 3 ; ++i)
    {
        unsigned int tmp = _brick_dim[i]/local_group_size[i];
        if(tmp*local_group_size[i] == _brick_dim[i])
        {
            global_group_size[i] = tmp;
        }
        else
        {
            global_group_size[i] = tmp + 1;
        }
    }
    glDispatchCompute(global_group_size[0] , global_group_size[1] , global_group_size[2]);

    _gl_program->unbind();

    CHECK_GL_ERROR;

#undef BRICK_VOLUME_INFO_BUFFER
#undef BRICK_SIZE
#undef BRICK_MARGIN
#undef BRICK_DIM
#undef VOLUME_DIM
#undef VOLUME_TEXTURE
#undef VOLUME_MIN_SCALAR
#undef VOLUME_REGULATE_PARAMETER
}

void VolumeBrickInfoCalculator::initialize_i()
{
    if (nullptr == _gl_program)
    {
        UIDType uid;
        _gl_program = GLResourceManagerContainer::instance()->get_program_manager()->create_object(uid);
        _gl_program->set_description("volume brick info calculator program");
        _gl_program->initialize();

        std::vector<GLShaderInfo> shaders;
        shaders.push_back(GLShaderInfo(GL_COMPUTE_SHADER , S_BRICK_INFO_CAL_VOLUME , "brick info cal volume compute shader"));
        shaders.push_back(GLShaderInfo(GL_COMPUTE_SHADER , S_BRICK_INFO_CAL_UTILS , "brick info cal utils compute shader"));
        _gl_program->set_shaders(shaders);
        _gl_program->compile();

        _res_shield.add_shield<GLProgram>(_gl_program);
    }
}

MaskBrickInfoCalculator::MaskBrickInfoCalculator()
{
}

MaskBrickInfoCalculator::~MaskBrickInfoCalculator()
{
}

void MaskBrickInfoCalculator::update(const AABBUI& aabb)
{
    initialize_i();
    calculate_i();
    update_i(aabb);
}


void MaskBrickInfoCalculator::calculate()
{
    initialize_i();
    calculate_i();
    download_i();
}

void MaskBrickInfoCalculator::download_i()
{
    _info_buffer->set_buffer_target(GL_SHADER_STORAGE_BUFFER);
    _info_buffer->bind();
    _info_buffer->download(_brick_dim[0]*_brick_dim[1]*_brick_dim[2]*sizeof(MaskBrickInfo) , _info_array);
    _info_buffer->unbind();
}

void MaskBrickInfoCalculator::calculate_i()
{
    AABBUI aabb;
    aabb._min[0] = 0;
    aabb._min[1] = 0;
    aabb._min[2] = 0;

    aabb._max[0] = _img_data->_dim[0];
    aabb._max[1] = _img_data->_dim[1];
    aabb._max[2] = _img_data->_dim[2];

    this->update_i(aabb);
}

void MaskBrickInfoCalculator::update_i(const AABBUI& aabb)
{
    RENDERALGO_CHECK_NULL_EXCEPTION(_img_data);
    RENDERALGO_CHECK_NULL_EXCEPTION(_img_texture);
    RENDERALGO_CHECK_NULL_EXCEPTION(_info_buffer);
    RENDERALGO_CHECK_NULL_EXCEPTION(_info_array);
    if (_visible_labels.empty())
    {
        RENDERALGO_THROW_EXCEPTION("empty visible label when calculate mask brick info!");
    }

#define BRICK_MASK_INFO_BUFFER 0
#define VISIBLE_LABEL_BUFFER 1
#define BRICK_SIZE 2
#define BRICK_MARGIN 3
#define BRICK_DIM 4
#define MASK_DIM 5
#define MASK_TEXTURE 6
#define BRICK_RANGE_MIN 7
#define BRICK_RANGE_DIM 8
#define VISIBLE_LABEL_COUNT 9

    CHECK_GL_ERROR;

    _gl_program->bind();

    const unsigned int program_id = _gl_program->get_id();

    glProgramUniform1i(program_id, BRICK_SIZE , _brick_size);
    glProgramUniform1i(program_id , BRICK_MARGIN , _brick_margin);
    glProgramUniform3i(program_id , BRICK_DIM , _brick_dim[0] , _brick_dim[1] , _brick_dim[2]);
    glProgramUniform3i(program_id , MASK_DIM ,static_cast<GLint>(_img_data->_dim[0]) , static_cast<GLint>(_img_data->_dim[1]) , static_cast<GLint>(_img_data->_dim[2]));

    _img_texture->bind();
    glActiveTexture(GL_TEXTURE0);
    GLTextureUtils::set_3d_wrap_s_t_r(GL_CLAMP_TO_BORDER);
    GLTextureUtils::set_filter(GL_TEXTURE_3D , GL_NEAREST);
    glProgramUniform1i(program_id , MASK_TEXTURE , 0);

    _info_buffer->bind_buffer_base(GL_SHADER_STORAGE_BUFFER , BRICK_MASK_INFO_BUFFER);

    glProgramUniform3i(program_id , BRICK_RANGE_MIN , aabb._min[0] , aabb._min[1] , aabb._min[2]);
    glProgramUniform3i(program_id , BRICK_RANGE_DIM , aabb._max[0] - aabb._min[0] , aabb._max[1] - aabb._min[1] , aabb._max[2] - aabb._min[2]);

    std::unique_ptr<int[]> visible_label_array(new int[_visible_labels.size()]);
    for (size_t i= 0; i<_visible_labels.size() ; ++i)
    {
        visible_label_array[i] = static_cast<int>(_visible_labels[i]);
    }
    _gl_buffer_visible_labels->bind();
    _gl_buffer_visible_labels->load(static_cast<GLsizei>(_visible_labels.size()), visible_label_array.get() , GL_STATIC_DRAW);
    _gl_buffer_visible_labels->bind_buffer_base(GL_SHADER_STORAGE_BUFFER , VISIBLE_LABEL_BUFFER);
    glProgramUniform1i(program_id , VISIBLE_LABEL_COUNT , static_cast<int>(_visible_labels.size()));

    const unsigned int local_group_size[3] = {8,8,1};//This parameter is optimal in ***
    unsigned int global_group_size[3] = {0};
    for (int i = 0; i< 3 ; ++i)
    {
        unsigned int tmp = _brick_dim[i]/local_group_size[i];
        if(tmp*local_group_size[i] == _brick_dim[i])
        {
            global_group_size[i] = tmp;
        }
        else
        {
            global_group_size[i] = tmp + 1;
        }
    }
    glDispatchCompute(global_group_size[0] , global_group_size[1] , global_group_size[2]);

    _gl_program->unbind();

    CHECK_GL_ERROR;

#undef BRICK_MASK_INFO_BUFFER
#undef VISIBLE_LABEL_BUFFER
#undef BRICK_SIZE
#undef BRICK_MARGIN
#undef BRICK_DIM
#undef MASK_DIM
#undef MASK_TEXTURE
#undef BRICK_RANGE_MIN
#undef BRICK_RANGE_DIM
#undef VISIBLE_LABEL_COUNT
}

void MaskBrickInfoCalculator::initialize_i()
{
    if (nullptr == _gl_program)
    {
        UIDType uid;
        _gl_program = GLResourceManagerContainer::instance()->get_program_manager()->create_object(uid);
        _gl_program->set_description("mask brick info calculator program");
        _gl_program->initialize();

        std::vector<GLShaderInfo> shaders;
        shaders.push_back(GLShaderInfo(GL_COMPUTE_SHADER , S_BRICK_INFO_CAL_MASK , "brick info cal mask compute shader"));
        shaders.push_back(GLShaderInfo(GL_COMPUTE_SHADER , S_BRICK_INFO_CAL_UTILS , "brick info cal utils compute shader"));
        _gl_program->set_shaders(shaders);
        _gl_program->compile();

        _gl_buffer_visible_labels = GLResourceManagerContainer::instance()->get_buffer_manager()->create_object(uid);
        _gl_buffer_visible_labels->set_description("visible label buffer count in mask brick info calculator program");
        _gl_buffer_visible_labels->initialize();
        _gl_buffer_visible_labels->set_buffer_target(GL_SHADER_STORAGE_BUFFER);

        _res_shield.add_shield<GLProgram>(_gl_program);
        _res_shield.add_shield<GLBuffer>(_gl_buffer_visible_labels);
    }
}

void MaskBrickInfoCalculator::set_visible_labels(const std::vector<unsigned char>& labels)
{
    _visible_labels = labels;
}

MED_IMG_END_NAMESPACE
