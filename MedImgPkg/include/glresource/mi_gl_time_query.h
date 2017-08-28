#ifndef MEDIMGRESOURCE_GL_TIME_QUERY_H_
#define MEDIMGRESOURCE_GL_TIME_QUERY_H_

#include "mi_gl_object.h"

MED_IMG_BEGIN_NAMESPACE 

class GLResource_Export GLTimeQuery : public GLObject {
public:
  GLTimeQuery(UIDType uid);

  ~GLTimeQuery();

  virtual void initialize();

  virtual void finalize();

  void get_id(unsigned int (&query_id)[2]) const;

  void begin();

  double end();

  double get_time_elapsed();

private:
  unsigned int _query[2];
  double _time_elapsed;
  bool _is_first_query;
};

MED_IMG_END_NAMESPACE

#endif
