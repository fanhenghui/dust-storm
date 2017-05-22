#ifndef MED_IMAGING_TEXTURE_1D_ARRAY_H
#define MED_IMAGING_TEXTURE_1D_ARRAY_H

#include "Core/gl_texture_base.h"



class  GLTexture1DArray : public GLTextureBase
{
public:
    GLTexture1DArray(UIDType uid);

    ~GLTexture1DArray();

    virtual void bind();

    virtual void unbind();

    void load(GLint internalformat , GLsizei width, GLsizei arraysize , GLenum format , GLenum type , 
        const void* data , GLint level = 0);

    void update(GLint xoffset , GLsizei width , GLsizei arrayidx, GLenum format , GLenum type , 
        const void* data , GLint level = 0);

    void download(GLenum format , GLenum type ,  void* buffer , GLint level = 0) const;

    int get_width();

    int get_array_size();

    GLenum get_format();

protected:
private:
    GLsizei _width;
    GLsizei _array_size;
    GLenum _format;
    GLenum _internal_format;
    GLenum _type;
};



#endif