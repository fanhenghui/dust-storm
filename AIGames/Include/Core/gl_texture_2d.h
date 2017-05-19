#ifndef MED_IMAGING_TEXTURE_2D_H
#define MED_IMAGING_TEXTURE_2D_H

#include "Core/gl_texture_base.h"



class  GLTexture2D : public GLTextureBase
{
public:
    GLTexture2D(UIDType uid);

    ~GLTexture2D();

    virtual void bind();

    virtual void unbind();

    void load(GLint internalformat , GLsizei width, GLsizei height , GLenum format , GLenum type , 
        const void* data , GLint level = 0);

    void update(GLint xoffset , GLint yoffset ,GLsizei width , GLsizei height , GLenum format , GLenum type , 
        const void* data , GLint level = 0);

    void download(GLenum format , GLenum type ,  void* buffer , GLint level = 0) const;

    GLsizei get_width();

    GLsizei get_height();

    GLenum get_format();

protected:
private:
    GLsizei _width;
    GLsizei _height;
    GLenum _format;
    GLenum _internal_format;
    GLenum _type;
};



#endif