#ifndef MEDIMGRESOURCE_GL_RESOURCE_MANAGER_CONTAINER_H
#define MEDIMGRESOURCE_GL_RESOURCE_MANAGER_CONTAINER_H

#include "glresource/mi_gl_resource_manager.h"
#include <list>

MED_IMG_BEGIN_NAMESPACE 

class GLProgram;
class GLBuffer;
class GLTexture1D;
class GLTexture2D;
class GLTexture3D;
class GLVAO;
class GLFBO;
class GLTexture1DArray;
class GLContext;
class GLTimeQuery;

typedef GLResourceManager<GLProgram> GLProgramManager;
typedef GLResourceManager<GLBuffer> GLBufferManager;
typedef GLResourceManager<GLTexture1D> GLTexture1DManager;
typedef GLResourceManager<GLTexture2D> GLTexture2DManager;
typedef GLResourceManager<GLTexture3D> GLTexture3DManager;
typedef GLResourceManager<GLVAO> GLVAOManager;
typedef GLResourceManager<GLFBO> GLFBOManager;
typedef GLResourceManager<GLTexture1DArray> GLTexture1DArrayManager;
typedef GLResourceManager<GLContext> GLContextManager;
typedef GLResourceManager<GLTimeQuery> GLTimeQueryManager;

typedef std::shared_ptr<GLProgramManager> GLProgramManagerPtr;
typedef std::shared_ptr<GLBufferManager> GLBufferManagerPtr;
typedef std::shared_ptr<GLTexture1DManager> GLTexture1DManagerPtr;
typedef std::shared_ptr<GLTexture2DManager> GLTexture2DManagerPtr;
typedef std::shared_ptr<GLTexture3DManager> GLTexture3DManagerPtr;
typedef std::shared_ptr<GLVAOManager> GLVAOManagerPtr;
typedef std::shared_ptr<GLFBOManager> GLFBOManagerPtr;
typedef std::shared_ptr<GLTexture1DArrayManager> GLTexture1DArrayManagerPtr;
typedef std::shared_ptr<GLContextManager> GLContextManagerPtr;
typedef std::shared_ptr<GLTimeQueryManager> GLTimeQueryManagerPtr;

class GLResource_Export GLResourceManagerContainer {
public:
  static GLResourceManagerContainer *instance();

  ~GLResourceManagerContainer();

  template <class ResourceType>
  std::shared_ptr<GLResourceManager<ResourceType>> get_resource_manager() {
    GLRESOURCE_THROW_EXCEPTION("get resource mananger failed!");
  }

  GLProgramManagerPtr get_program_manager() const;

  GLBufferManagerPtr get_buffer_manager() const;

  GLTexture1DManagerPtr get_texture_1d_manager() const;

  GLTexture2DManagerPtr get_texture_2d_manager() const;

  GLTexture3DManagerPtr get_texture_3d_manager() const;

  GLTexture1DArrayManagerPtr get_texture_1d_array_manager() const;

  GLVAOManagerPtr get_vao_manager() const;

  GLFBOManagerPtr get_fbo_manager() const;

  GLContextManagerPtr get_context_manager() const;

  GLTimeQueryManagerPtr get_time_query_manager() const;

  void update_all();

private:
  GLResourceManagerContainer();

private:
  static GLResourceManagerContainer *_s_instance;
  static boost::mutex _mutex;

private:
  GLProgramManagerPtr _program_manager;
  GLBufferManagerPtr _buffer_manager;
  GLTexture1DManagerPtr _texture_1d_manager;
  GLTexture2DManagerPtr _texture_2d_manager;
  GLTexture3DManagerPtr _texture_3d_manager;
  GLTexture1DArrayManagerPtr _texture_1d_array_manager;
  GLVAOManagerPtr _vao_manager;
  GLFBOManagerPtr _fbo_manager;
  GLContextManagerPtr _context_manager;
  GLTimeQueryManagerPtr _time_query_manager;
};

class GLObjectShieldBase {
public:
  GLObjectShieldBase(){};
  virtual ~GLObjectShieldBase(){};

  virtual bool shield(std::shared_ptr<GLObject> obj) = 0;
};

template <class ResourceType> class GLObjectShield : public GLObjectShieldBase {
public:
  GLObjectShield(std::shared_ptr<ResourceType> obj) : _obj(obj){};

  virtual ~GLObjectShield() {
    if (_obj) {
      GLResourceManagerContainer::instance()
          ->get_resource_manager<ResourceType>()
          ->remove_object(_obj->get_uid());
    }
  };

  virtual bool shield(std::shared_ptr<GLObject> obj) {
    if (!_obj) {
      return false;
    }
    return obj == _obj;
  }

private:
  std::shared_ptr<ResourceType> _obj;
};

class GLResourceShield {
public:
  GLResourceShield(){};

  ~GLResourceShield() {
    for (auto it = _shields.begin(); it != _shields.end(); ++it) {
      delete *it;
    }
    _shields.clear();
  }

  template <class ResourceType>
  void add_shield(std::shared_ptr<ResourceType> obj) {
    _shields.push_back(new GLObjectShield<ResourceType>(obj));
  }

  template <class ResourceType>
  void remove_shield(std::shared_ptr<ResourceType> obj) {
    for (auto it = _shields.begin(); it != _shields.end(); ++it) {
      if ((*it)->shield(obj)) {
        _shields.erase(it);
        break;
      }
    }
  }

private:
  std::list<GLObjectShieldBase *> _shields;
};

MED_IMG_END_NAMESPACE

#endif
