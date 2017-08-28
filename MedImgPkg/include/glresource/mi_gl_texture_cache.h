#ifndef MEDIMGGLRESOURCE_TEXTURE_CACHE_H
#define MEDIMGGLRESOURCE_TEXTURE_CACHE_H

#include <list>
#include "GL/glew.h"
#include "boost/thread/mutex.hpp"
#include "glresource/mi_gl_resource_export.h"
#include "glresource/mi_gl_resource_define.h"

MED_IMG_BEGIN_NAMESPACE 

class GLResource_Export GLTextureCache {
public:
    static GLTextureCache* instance();

    ~GLTextureCache();

    //upload
    void cache_load(GLenum target , GLTextureBasePtr texture , 
        GLint wrap_type , GLint filter_type , 
        GLint internalformat , 
        GLsizei width, GLsizei height , GLsizei depth , 
        GLenum format , GLenum type , char* data , GLint level = 0);

    //update
    void cache_update(GLenum target , GLTextureBasePtr texture , 
        GLint xoffset , GLint yoffset , GLint zoffset ,
        GLsizei width , GLsizei height , GLsizei depth , 
        GLenum format , GLenum type , char* data , GLint level = 0);

    //process in gpu
    void process_cache();

private:
    GLTextureCache();

    static GLTextureCache* _s_instance;
    static boost::mutex _s_mutex;

    boost::mutex _mutex;
    struct TextureUnit;
    std::list<std::shared_ptr<TextureUnit>> _cache_list;

private:
};

MED_IMG_END_NAMESPACE
#endif