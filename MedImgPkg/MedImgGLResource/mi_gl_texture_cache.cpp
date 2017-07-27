#include "mi_gl_texture_cache.h"

#include "mi_gl_texture_1d.h"
#include "mi_gl_texture_2d.h"
#include "mi_gl_texture_3d.h"

MED_IMG_BEGIN_NAMESPACE

struct GLTextureCache::TextureUnit
{
    GLenum target;
    GLTextureBasePtr texture;
    GLint internalformat;
    GLint xoffset;
    GLint yoffset;
    GLint zoffset;
    GLsizei width;
    GLsizei height;
    GLsizei depth ;
    GLenum format;
    GLenum type;
    char* data;
    GLint level;

    int cache_type;//0 for load 1 for update

    TextureUnit()
    {
        target = GL_TEXTURE_1D;
        internalformat = GL_RGB8;
        xoffset = 0;
        yoffset = 0;
        zoffset = 0;
        width = 0;
        height = 0;
        depth = 0;
        format = GL_RGB;
        type = GL_UNSIGNED_BYTE;
        data = nullptr;
        level = 0;

        cache_type = 0;
    }

    ~TextureUnit()
    {
        if (nullptr != data)
        {
            delete [] data;
            data = nullptr;
        }
    }
};

boost::mutex GLTextureCache::_s_mutex;

GLTextureCache* GLTextureCache::_s_instance = nullptr;

GLTextureCache* GLTextureCache::instance() {
    if (!_s_instance) {
        boost::unique_lock<boost::mutex> locker(_s_mutex);

        if (!_s_instance) {
            _s_instance = new GLTextureCache();
        }
    }

    return _s_instance;
}

GLTextureCache::GLTextureCache()
{

}

GLTextureCache::~GLTextureCache() {

}

void GLTextureCache::cache(
    GLenum target , GLTextureBasePtr texture , GLint internalformat , 
    GLsizei width, GLsizei height , GLsizei depth , 
    GLenum format , GLenum type , char* data , GLint level /*= 0*/)
{
    boost::mutex::scoped_lock lock(_mutex);

    TextureUnit unit;
    unit.cache_type = 0;

    unit.target= target;
    unit.texture= texture;
    unit.internalformat= internalformat;
    unit.width= width;
    unit.height= height;
    unit.depth= depth;
    unit.format= format;
    unit.type= type;
    unit.data= data;
    unit.level = level;

    _cache_list.push_back(unit);
}

void GLTextureCache::cache(
    GLenum target , GLTextureBasePtr texture , 
    GLint xoffset , GLint yoffset , GLint zoffset ,
    GLsizei width , GLsizei height , GLsizei depth , 
    GLenum format , GLenum type , char* data , GLint level /*= 0*/)
{
    boost::mutex::scoped_lock lock(_mutex);

    TextureUnit unit;
    unit.cache_type = 1;

    unit.target= target;
    unit.texture= texture;
    unit.xoffset= xoffset;
    unit.yoffset= yoffset;
    unit.zoffset= zoffset;
    unit.width= width;
    unit.height= height;
    unit.depth= depth;
    unit.format= format;
    unit.type= type;
    unit.data= data;
    unit.level = level;

    _cache_list.push_back(unit);
}

void GLTextureCache::process_cache()
{
    boost::mutex::scoped_lock lock(_mutex);

    for (auto it = _cache_list.begin() ; it != _cache_list.end() ; ++it)
    {
        TextureUnit& unit = *it;
        switch(unit.target)
        {
        case GL_TEXTURE_1D:
            {
                GLTexture1DPtr cache_tex1d = std::dynamic_pointer_cast<GLTexture1D>(unit.texture);
                GLRESOURCE_CHECK_NULL_EXCEPTION(cache_tex1d);
                if (0 == unit.cache_type )
                {
                    cache_tex1d->load(unit.internalformat , unit.width , unit.format , unit.type , unit.data , unit.level);
                }
                else if(1 ==  unit.cache_type)
                {
                    cache_tex1d->update(unit.xoffset , unit.width , unit.format , unit.type , unit.data , unit.level);
                }
                break;
            }
        case GL_TEXTURE_2D:
            {
                GLTexture2DPtr cache_tex2d = std::dynamic_pointer_cast<GLTexture2D>(unit.texture);
                GLRESOURCE_CHECK_NULL_EXCEPTION(cache_tex2d);
                if (0 == unit.cache_type )
                {
                    cache_tex2d->load(unit.internalformat , unit.width , unit.height  , unit.format , unit.type , unit.data , unit.level);
                }
                else if(1 ==  unit.cache_type)
                {
                    cache_tex2d->update(unit.xoffset , unit.yoffset , unit.width ,unit.height , unit.format , unit.type , unit.data , unit.level);
                }
                break;
            }
        case GL_TEXTURE_3D:
            {
                GLTexture3DPtr cache_tex3d = std::dynamic_pointer_cast<GLTexture3D>(unit.texture);
                GLRESOURCE_CHECK_NULL_EXCEPTION(cache_tex3d);
                if (0 == unit.cache_type )
                {
                    cache_tex3d->load(unit.internalformat , unit.width , unit.height , unit.depth , unit.format , unit.type , unit.data , unit.level);
                }
                else if(1 ==  unit.cache_type)
                {
                    cache_tex3d->update(unit.xoffset , unit.yoffset , unit.zoffset , unit.width , unit.height , unit.depth ,unit.format , unit.type , unit.data , unit.level);
                }
                break;
            }
        default:
            {
                GLRESOURCE_THROW_EXCEPTION("Invalid texture target in cache!");
            }
        }
    }

    _cache_list.clear();
}


MED_IMG_END_NAMESPACE
