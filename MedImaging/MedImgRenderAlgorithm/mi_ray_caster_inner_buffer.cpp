#include "mi_ray_caster_inner_buffer.h"
#include "MedImgGLResource/mi_gl_utils.h"
#include "MedImgGLResource/mi_gl_resource_manager_container.h"
#include "MedImgGLResource/mi_gl_buffer.h"

MED_IMAGING_BEGIN_NAMESPACE

struct RayCasterInnerBuffer::GLResource
{
    std::map<BufferType , GLBufferPtr> buffer_ids;
    bool dirty_flag[TYPE_END];

    GLResource()
    {
        memset(dirty_flag , 1 , sizeof(bool)*TYPE_END);
    }

    void release()
    {
        for (auto it = buffer_ids.begin() ; it != buffer_ids.end() ; ++it)
        {
            GLResourceManagerContainer::instance()->get_buffer_manager()->remove_object(it->second->get_uid());
        }
        buffer_ids.clear();
        memset(dirty_flag , 0 , sizeof(bool)*TYPE_END);
    }

    GLBufferPtr GetBuffer(BufferType type)
    {
        auto it = buffer_ids.find(type);
        if (it != buffer_ids.end())
        {
            return it->second;
        }
        else
        {
            UIDType buffer_id= 0;
            GLBufferPtr buffer = GLResourceManagerContainer::instance()->get_buffer_manager()->create_object(buffer_id);
            buffer->initialize();
            buffer->set_buffer_target(GL_SHADER_STORAGE_BUFFER);
            buffer_ids[type] = buffer;
            return buffer;
        }
    }
};

RayCasterInnerBuffer::RayCasterInnerBuffer():_inner_resource(new GLResource())
{

}

RayCasterInnerBuffer::~RayCasterInnerBuffer()
{
    release_buffer();
}

void RayCasterInnerBuffer::release_buffer()
{
    _inner_resource->release();
}

GLBufferPtr RayCasterInnerBuffer::get_buffer(BufferType type)
{
    try
    {
        GLBufferPtr buffer = _inner_resource->GetBuffer(type);

        CHECK_GL_ERROR;

        switch(type)
        {
        case WINDOW_LEVEL_BUCKET:
            {
                if (_inner_resource->dirty_flag[WINDOW_LEVEL_BUCKET])
                {
                    float* wl_array = (float*)_shared_buffer_array.get();
                    memset(wl_array , 0 , sizeof(float)*static_cast<int>(_label_level)*2);

                    for (auto it = _window_levels.begin() ; it != _window_levels.end() ; ++it)
                    {
                        const unsigned char label = it->first;
                        if (label > static_cast<int>(_label_level) - 1)
                        {
                            std::stringstream ss;
                            ss << "Input window level label : " << (int)(label) << " is greater than the limit : " << static_cast<int>(_label_level) - 1 << " !";
                            RENDERALGO_THROW_EXCEPTION(ss.str());
                        }
                        wl_array[label*2] = it->second._value.x;
                        wl_array[label*2+1] = it->second._value.y;
                    }

                    buffer->bind();
                    buffer->load(static_cast<int>(_label_level)*sizeof(float)*2 , wl_array, GL_DYNAMIC_DRAW);

                    _inner_resource->dirty_flag[WINDOW_LEVEL_BUCKET] = false;
                }
                break;
            }

        case VISIBLE_LABEL_BUCKET:
            {
                if (_inner_resource->dirty_flag[VISIBLE_LABEL_BUCKET])
                {
                    int* label_array = (int*)_shared_buffer_array.get();
                    memset(label_array , 0 , sizeof(int)*static_cast<int>(_label_level));

                    for (auto it = _labels.begin() ; it != _labels.end() ; ++it)
                    {
                        if (*it > static_cast<int>(_label_level) - 1)
                        {
                            std::stringstream ss;
                            ss << "Input visible label : " << (int)(*it ) << " is greater than the limit : " << static_cast<int>(_label_level) - 1 << " !";
                            RENDERALGO_THROW_EXCEPTION(ss.str());
                        }
                        label_array[*it] = 1;
                    }

                    buffer->bind();
                    buffer->load(static_cast<int>(_label_level)*sizeof(int) , label_array , GL_STATIC_DRAW);

                    _inner_resource->dirty_flag[VISIBLE_LABEL_BUCKET] = false;
                }
                break;
            }

        default:
            {
                RENDERALGO_THROW_EXCEPTION("Invalid buffer type!");
            }
        }

        CHECK_GL_ERROR;

        return buffer;

    }
    catch (Exception& e)
    {
#ifdef _DEBUG
        std::cout << e.what();
#endif
        assert(false);
        throw e;
    }
}

void RayCasterInnerBuffer::set_mask_label_level(LabelLevel eLevel)
{
    if (_label_level != eLevel)
    {
        _label_level = eLevel;
        memset(_inner_resource->dirty_flag , 1 , sizeof(bool)*TYPE_END);
        _shared_buffer_array.reset(new char[static_cast<int>(_label_level)*2*4]);
    }
}

void RayCasterInnerBuffer::set_window_level(float ww , float wl , unsigned char label)
{
    const Vector2f wl_v2(ww , wl);
    auto it = _window_levels.find(label);
    if (it == _window_levels.end())
    {
        _window_levels.insert(std::make_pair(label ,wl_v2));
        _inner_resource->dirty_flag[WINDOW_LEVEL_BUCKET] = true;
    }
    else
    {
        if (wl_v2 != it->second)
        {
            it->second = wl_v2;
            _inner_resource->dirty_flag[WINDOW_LEVEL_BUCKET] = true;
        }
    }
}

void RayCasterInnerBuffer::set_visible_labels(std::vector<unsigned char> labels)
{
    if (_labels != labels)
    {
        _labels =labels;
        _inner_resource->dirty_flag[VISIBLE_LABEL_BUCKET] = true;
    }
}

MED_IMAGING_END_NAMESPACE