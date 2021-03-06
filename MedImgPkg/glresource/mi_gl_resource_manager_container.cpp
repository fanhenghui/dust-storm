#include "mi_gl_resource_manager_container.h"
#include "mi_gl_buffer.h"
#include "mi_gl_context.h"
#include "mi_gl_fbo.h"
#include "mi_gl_program.h"
#include "mi_gl_texture_1d.h"
#include "mi_gl_texture_1d_array.h"
#include "mi_gl_texture_2d.h"
#include "mi_gl_texture_3d.h"
#include "mi_gl_time_query.h"
#include "mi_gl_vao.h"

#include "mi_gl_resource_logger.h"

MED_IMG_BEGIN_NAMESPACE

#ifndef WIN32
#include <stdio.h>

static int x_error_handler(Display* dpy, XErrorEvent *event) {
    std::stringstream ss;

    char buffer[BUFSIZ];
    char mesg[BUFSIZ];
    char number[32];
    const char *mtype = "XlibMessage";

    XGetErrorText(dpy, event->error_code, buffer, BUFSIZ);
    XGetErrorDatabaseText(dpy, mtype, "XError", "X Error", mesg, BUFSIZ);
    ss << "catch " << mesg << ": " << buffer << ". ";
    XGetErrorDatabaseText(dpy, mtype, "MajorCode", "Request Major code %d", mesg, BUFSIZ);

    ss << "Failed request status: " << "{ ";
    ss << "Major opcode: " <<  (int)event->request_code;
    if (event->request_code < 128) {
	    snprintf(number, sizeof(number), "%d", event->request_code);
        XGetErrorDatabaseText(dpy, "XRequest", number, "", buffer, BUFSIZ);
        ss << "(" << buffer << ")";
    }
    ss << ", ";
    
    ss << "Minor opcode: " <<  (int)event->minor_code << ", ";
    ss << "Error code: " <<  (int)event->error_code << ", ";
    ss << "Serial number: " <<  (int)event->serial << "}. ";

    MI_GLRESOURCE_LOG(MI_ERROR) << ss.str();
    return 0;
}

#endif

boost::mutex GLResourceManagerContainer::_mutex;

GLResourceManagerContainer* GLResourceManagerContainer::_s_instance = nullptr;

GLResourceManagerContainer* GLResourceManagerContainer::instance() {
    if (nullptr == _s_instance) {
        boost::unique_lock<boost::mutex> locker(_mutex);

        if (nullptr == _s_instance) {
            _s_instance = new GLResourceManagerContainer();
        }
    }

    return _s_instance;
}

GLResourceManagerContainer::~GLResourceManagerContainer() {}

void GLResourceManagerContainer::update_all() {
    _program_manager->update();
    _buffer_manager->update();
    _texture_1d_manager->update();
    _texture_2d_manager->update();
    _texture_3d_manager->update();
    _vao_manager->update();
    _fbo_manager->update();
    _texture_1d_array_manager->update();
    _context_manager->update();
    _time_query_manager->update();
}

std::string GLResourceManagerContainer::get_specification(const std::string& split /*= " "*/) {
    std::stringstream ss;
    ss << "GL Resources: [" << split <<
        _program_manager->get_specification(split) << ", " << split <<
        _buffer_manager->get_specification(split) << ", " << split <<
        _texture_1d_manager->get_specification(split) << ", " << split <<
        _texture_2d_manager->get_specification(split) << ", " << split <<
        _texture_3d_manager->get_specification(split) << ", " << split <<
        _texture_1d_array_manager->get_specification(split) << ", " << split <<
        _vao_manager->get_specification(split) << ", " << split <<
        _fbo_manager->get_specification(split) << ", " << split <<
        _context_manager->get_specification(split) << ", " << split <<
        _time_query_manager->get_specification(split) << ", " << split << "]";
    return ss.str();
}

GLResourceManagerContainer::GLResourceManagerContainer()
    : _program_manager(new GLProgramManager()),
      _buffer_manager(new GLBufferManager()),
      _texture_1d_manager(new GLTexture1DManager()),
      _texture_2d_manager(new GLTexture2DManager()),
      _texture_3d_manager(new GLTexture3DManager()),
      _texture_1d_array_manager(new GLTexture1DArrayManager()),
      _vao_manager(new GLVAOManager()),
      _fbo_manager(new GLFBOManager()),
      _context_manager(new GLContextManager()),
      _time_query_manager(new GLTimeQueryManager()) {
#ifndef WIN32
    XSetErrorHandler(x_error_handler);
#endif
}

GLProgramManagerPtr GLResourceManagerContainer::get_program_manager() const {
    return _program_manager;
}

GLBufferManagerPtr GLResourceManagerContainer::get_buffer_manager() const {
    return _buffer_manager;
}

GLTexture1DManagerPtr
GLResourceManagerContainer::get_texture_1d_manager() const {
    return _texture_1d_manager;
}

GLTexture2DManagerPtr
GLResourceManagerContainer::get_texture_2d_manager() const {
    return _texture_2d_manager;
}

GLTexture3DManagerPtr
GLResourceManagerContainer::get_texture_3d_manager() const {
    return _texture_3d_manager;
}

GLVAOManagerPtr GLResourceManagerContainer::get_vao_manager() const {
    return _vao_manager;
}

GLFBOManagerPtr GLResourceManagerContainer::get_fbo_manager() const {
    return _fbo_manager;
}

GLTexture1DArrayManagerPtr
GLResourceManagerContainer::get_texture_1d_array_manager() const {
    return _texture_1d_array_manager;
}

GLContextManagerPtr GLResourceManagerContainer::get_context_manager() const {
    return _context_manager;
}

GLTimeQueryManagerPtr
GLResourceManagerContainer::get_time_query_manager() const {
    return _time_query_manager;
}

//gcc and msvc's template specialization are different.
//gcc : write out of class define(in cpp)
//msvc write in class(in h)
#ifndef WIN32
template <>
std::shared_ptr<GLResourceManager<GLProgram>>
    GLResourceManagerContainer::get_resource_manager<GLProgram>() {
        return _program_manager;
};
template <>
std::shared_ptr<GLResourceManager<GLBuffer>>
    GLResourceManagerContainer::get_resource_manager<GLBuffer>() {
        return _buffer_manager;
};
template <>
std::shared_ptr<GLResourceManager<GLTexture1D>>
    GLResourceManagerContainer::get_resource_manager<GLTexture1D>() {
        return _texture_1d_manager;
};
template <>
std::shared_ptr<GLResourceManager<GLTexture1DArray>>
    GLResourceManagerContainer::get_resource_manager<GLTexture1DArray>() {
        return _texture_1d_array_manager;
};
template <>
std::shared_ptr<GLResourceManager<GLTexture2D>>
    GLResourceManagerContainer::get_resource_manager<GLTexture2D>() {
        return _texture_2d_manager;
};
template <>
std::shared_ptr<GLResourceManager<GLTexture3D>>
    GLResourceManagerContainer::get_resource_manager<GLTexture3D>() {
        return _texture_3d_manager;
};
template <>
std::shared_ptr<GLResourceManager<GLVAO>>
    GLResourceManagerContainer::get_resource_manager<GLVAO>() {
        return _vao_manager;
};
template <>
std::shared_ptr<GLResourceManager<GLFBO>>
    GLResourceManagerContainer::get_resource_manager<GLFBO>() {
        return _fbo_manager;
};
template <>
std::shared_ptr<GLResourceManager<GLContext>>
    GLResourceManagerContainer::get_resource_manager<GLContext>() {
        return _context_manager;
};
template <>
std::shared_ptr<GLResourceManager<GLTimeQuery>>
    GLResourceManagerContainer::get_resource_manager<GLTimeQuery>() {
        return _time_query_manager;
};

#endif


MED_IMG_END_NAMESPACE