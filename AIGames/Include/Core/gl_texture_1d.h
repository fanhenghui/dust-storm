#ifndef MED_IMAGING_TEXTURE_1D_H
#define MED_IMAGING_TEXTURE_1D_H

#include "Core/gl_texture_base.h"



class  GLTexture1D : public GLTextureBase
{
public:
    GLTexture1D(UIDType uid);

    ~GLTexture1D();

    virtual void bind();

    virtual void unbind();

    void load(GLint internalformat , GLsizei width, GLenum format , GLenum type , 
        const void* data , GLint level = 0);

    void update(GLint xoffset , GLsizei width , GLenum format , GLenum type , 
        const void* data , GLint level = 0);

    void download(GLenum format , GLenum type ,  void* buffer , GLint level = 0) const;

    GLsizei get_width();

    GLenum get_format();

protected:
private:
    GLsizei _width;
    GLenum _format;
    GLenum _internal_format;
    GLenum _type;
};



#endif