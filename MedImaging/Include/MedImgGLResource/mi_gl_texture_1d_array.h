#ifndef MED_IMAGING_TEXTURE_1D_ARRAY_H
#define MED_IMAGING_TEXTURE_1D_ARRAY_H

#include "MedImgGLResource/mi_gl_texture_base.h"

MED_IMAGING_BEGIN_NAMESPACE

class GLResource_Export GLTexture1DArray : public GLTextureBase
{
public:
    GLTexture1DArray(UIDType uid);

    ~GLTexture1DArray();

    virtual void Bind();

    virtual void UnBind();

    void Load(GLint internalformat , GLsizei width, GLsizei arraysize , GLenum format , GLenum type , 
        const void* data , GLint level = 0);

    void Update(GLint xoffset , GLsizei width , GLsizei arrayidx, GLenum format , GLenum type , 
        const void* data , GLint level = 0);

    void Download(GLenum format , GLenum type ,  void* buffer , GLint level = 0) const;

    int GetWidth();

    int GetArraySize();

    GLenum GetFormat();

protected:
private:
    GLsizei m_iWidth;
    GLsizei m_iArraySize;
    GLenum m_eFormat;
    GLenum m_eInternalFormat;
    GLenum m_eType;
};

MED_IMAGING_END_NAMESPACE

#endif