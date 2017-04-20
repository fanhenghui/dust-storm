#ifndef MED_IMAGING_TEXTURE_2D_H
#define MED_IMAGING_TEXTURE_2D_H

#include "MedImgGLResource/mi_gl_texture_base.h"

MED_IMAGING_BEGIN_NAMESPACE

class GLResource_Export GLTexture2D : public GLTextureBase
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
    GLsizei m_iWidth;
    GLsizei m_iHeight;
    GLenum m_eFormat;
    GLenum m_eInternalFormat;
    GLenum m_eType;
};

MED_IMAGING_END_NAMESPACE

#endif