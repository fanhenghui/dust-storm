#ifndef MED_IMAGING_TEXTURE_1D_H
#define MED_IMAGING_TEXTURE_1D_H

#include "MedImgGLResource/mi_gl_texture_base.h"

MED_IMAGING_BEGIN_NAMESPACE

class GLResource_Export GLTexture1D : public GLTextureBase
{
public:
    GLTexture1D(UIDType uid);

    ~GLTexture1D();

    virtual void Bind();

    virtual void UnBind();

    void Load(GLint internalformat , GLsizei width, GLenum format , GLenum type , 
        const void* data , GLint level = 0);

    void Update(GLint xoffset , GLsizei width , GLenum format , GLenum type , 
        const void* data , GLint level = 0);

    void Download(GLenum format , GLenum type ,  void* buffer , GLint level = 0) const;

    GLsizei GetWidth();

    GLenum GetFormat();

protected:
private:
    GLsizei m_iWidth;
    GLenum m_eFormat;
    GLenum m_eInternalFormat;
    GLenum m_eType;
};

MED_IMAGING_END_NAMESPACE

#endif