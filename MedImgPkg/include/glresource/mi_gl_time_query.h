#ifndef MEDIMGRESOURCE_GL_TIME_QUERY_H_
#define MEDIMGRESOURCE_GL_TIME_QUERY_H_

#include "mi_gl_object.h"

MED_IMG_BEGIN_NAMESPACE

class GLResource_Export GLTimeQuery : public GLObject {
public:
    explicit GLTimeQuery(UIDType uid);

    ~GLTimeQuery();

    virtual void initialize();

    virtual void finalize();

    void get_id(unsigned int (&query_id)[2]) const;

    void begin();

    float end();

    float get_time_elapsed() const;

private:
    unsigned int _query[2];
    float _time_elapsed;
    bool _is_first_query;
};

class GLResource_Export ScopedGLTimeQuery
{
public:
    ScopedGLTimeQuery(std::shared_ptr<GLTimeQuery> tq, float* recorder) :_time_query(tq), _recorder(recorder) {
        if (_time_query) {
            _time_query->begin();
        }
    }

    ~ScopedGLTimeQuery() {
        if (_time_query) {
            if (_recorder) {
                *_recorder = _time_query->end();
            }
        }
    }
private:
    std::shared_ptr<GLTimeQuery> _time_query;
    float* _recorder;

private:
    DISALLOW_COPY_AND_ASSIGN(ScopedGLTimeQuery);
};

MED_IMG_END_NAMESPACE

#endif
