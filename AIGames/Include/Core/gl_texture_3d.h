#ifndef MED_IMAGING_TEXTURE_3D_H
#define MED_IMAGING_TEXTURE_3D_H

#include "Core/gl_texture_base.h"



class  GLTexture3D : public GLTextureBase
{
public:
    GLTexture3D(UIDType uid);

    ~GLTexture3D();

    virtual void bind();

    virtual void unbind();

    void load(GLint internalformat , GLsizei width, GLsizei height , GLsizei depth , GLenum format , GLenum type , 
        const void* data , GLint level = 0);

    void update(GLint xoffset , GLint yoffset , GLint zoffset ,GLsizei width , GLsizei height , GLsizei depth , GLenum format , GLenum type , 
        const void* data , GLint level = 0);

    void download(GLenum format , GLenum type ,  void* buffer , GLint level = 0) const;

    GLsizei get_width();

    GLsizei get_height();

    GLsizei get_depth();

    GLenum get_format();

protected:
private:
    GLsizei _width;
    GLsizei _height;
    GLsizei _depth;
    GLenum _format;
    GLenum _internal_format;
    GLenum _type;
};



#endif