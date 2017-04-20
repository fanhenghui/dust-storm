#ifndef MED_IMAGING_TEXTURE_1D_H
#define MED_IMAGING_TEXTURE_1D_H

#include "MedImgGLResource/mi_gl_texture_base.h"

MED_IMAGING_BEGIN_NAMESPACE

class GLResource_Export GLTexture1D : public GLTextureBase
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
    GLsizei m_iWidth;
    GLenum m_eFormat;
    GLenum m_eInternalFormat;
    GLenum m_eType;
};

MED_IMAGING_END_NAMESPACE

#endif