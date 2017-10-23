#ifndef MEDIMGRESOURCE_GL_RESOUECE_DEFINE_H_
#define MEDIMGRESOURCE_GL_RESOUECE_DEFINE_H_

#include <memory>
#include "glresource/mi_gl_resource_export.h"

MED_IMG_BEGIN_NAMESPACE

class GLTextureBase;
class GLTexture1D;
class GLTexture2D;
class GLTexture3D;
class GLTexture1DArray;
class GLBuffer;
class GLVAO;
class GLFBO;
class GLProgram;
class GLTimeQuery;

typedef std::shared_ptr<GLTextureBase> GLTextureBasePtr;
typedef std::shared_ptr<GLTexture1D> GLTexture1DPtr;
typedef std::shared_ptr<GLTexture2D> GLTexture2DPtr;
typedef std::shared_ptr<GLTexture3D> GLTexture3DPtr;
typedef std::shared_ptr<GLTexture1DArray> GLTexture1DArrayPtr;
typedef std::shared_ptr<GLBuffer> GLBufferPtr;
typedef std::shared_ptr<GLVAO> GLVAOPtr;
typedef std::shared_ptr<GLFBO> GLFBOPtr;
typedef std::shared_ptr<GLProgram> GLProgramPtr;
typedef std::shared_ptr<GLTimeQuery> GLTimeQueryPtr;

MED_IMG_END_NAMESPACE
#endif