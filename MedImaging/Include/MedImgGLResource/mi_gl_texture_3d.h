#ifndef MED_IMAGING_TEXTURE_3D_H
#define MED_IMAGING_TEXTURE_3D_H

#include "MedImgGLResource/mi_gl_texture_base.h"

MED_IMAGING_BEGIN_NAMESPACE

class GLResource_Export GLTexture3D : public GLTextureBase
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
    GLsizei m_iWidth;
    GLsizei m_iHeight;
    GLsizei m_iDepth;
    GLenum m_eFormat;
    GLenum m_eInternalFormat;
    GLenum m_eType;
};

MED_IMAGING_END_NAMESPACE

#endif