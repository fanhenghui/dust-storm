#include "mi_gl_time_query.h"

MED_IMG_BEGIN_NAMESPACE

GLTimeQuery::GLTimeQuery(UIDType uid):GLObject(uid),_is_first_query(true)
{
    _query[0] = 0;
    _query[1] = 0;
}

GLTimeQuery::~GLTimeQuery()
{

}

void GLTimeQuery::initialize()
{
    if(_query[0] == 0){
        glGenQueries(2 , _query);
    }
}

void GLTimeQuery::finalize()
{
    if(_query[0] != 0){
        glDeleteQueries(2 , _query);
        _query[0] = 0;
        _query[1] = 0;
    }
}

void GLTimeQuery::get_id(unsigned int (&query_id)[2]) const
{
    query_id[0] = _query[0];
    query_id[1] = _query[1];
}

void GLTimeQuery::begin()
{
    if(_query[0] == 0){
        GLRESOURCE_THROW_EXCEPTION("time query begin without initialize!");
    }
    if(!_is_first_query){
        //GLint time;
        //glGetQueryiv(GL_TIME_ELAPSED, GL_CURRENT_QUERY , &time);

    }

    glBeginQuery(GL_TIME_ELAPSED , _query[0]);
}

double GLTimeQuery::end()
{
    if(_query[0] == 0){
        GLRESOURCE_THROW_EXCEPTION("time query begin without initialize!");
    }

    glEndQuery(GL_TIME_ELAPSED);

    unsigned int done;
    glGetQueryObjectuiv(_query[1] , GL_QUERY_RESULT_AVAILABLE , &done);
    unsigned int time;
    glGetQueryObjectuiv(_query[1] , GL_QUERY_RESULT , &time);
    _time_elapsed = time / 1000000.0;

    std::swap(_query[0] , _query[1]);

    return _time_elapsed;
}

double GLTimeQuery::get_time_elapsed()
{
    return _time_elapsed;
}

MED_IMG_END_NAMESPACE