#ifndef MED_IMAGING_GL_ENVIRONMENT_H_
#define MED_IMAGING_GL_ENVIRONMENT_H_





class  GLEnvironment
{
public:
    GLEnvironment();

    ~GLEnvironment();

    //OpenGL 版本信息
    void get_gl_version(int &major , int &minor);

    //OpenGL 的实现厂商
    std::string get_gl_vendor();

    //OpenGL 渲染器（硬件平台）
    std::string get_gl_renderer();

protected:

private:
};



#endif