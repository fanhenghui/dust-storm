#ifndef MED_IMAGING_TEXTURE_BASE_H_
#define MED_IMAGING_TEXTURE_BASE_H_

#include "Core/gl_object.h"



class  GLTextureBase : public GLObject
{
public:
    GLTextureBase(UIDType uid);

    virtual ~GLTextureBase();

    virtual void initialize();

    virtual void finalize();

    unsigned int get_id() const;

    virtual void bind() = 0;

    //////////////////////////////////////////////////////////////////////////
    //bind a level of a texture to an image unit
    //unit         Specifies the index of the image unit to which to bind the texture
    //level        Specifies the level of the texture that is to be bound.
    //layered   Specifies whether a layered texture binding is to be established.
    //layer        If layered is GL_FALSE, specifies the layer of texture to be bound to the image unit. Ignored otherwise. 
    //                  EG , bind GL_TEXTURE_3D should set layered true , bind GL_TEXTURE_2D should not
    //access     Specifies a token indicating the type of access that will be performed on the image.
    //format     Specifies the format that the elements of the image will be treated as for the purposes of formatted stores.
    virtual void bind_image(GLuint unit , GLint level , GLboolean layered , GLint layer ,  GLenum access, GLenum format );

    virtual void unbind() = 0;

protected:
    unsigned int _texture_id;
};



#endif
