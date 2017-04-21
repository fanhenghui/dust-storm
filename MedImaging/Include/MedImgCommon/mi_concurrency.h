#ifndef MED_IMAGING_CONCURRENCY_H_
#define MED_IMAGING_CONCURRENCY_H_

#include "MedImgCommon/mi_common_stdafx.h"
#include "boost/thread/mutex.hpp"

MED_IMAGING_BEGIN_NAMESPACE

class Common_Export Concurrency
{
public:
    static Concurrency* instance();
    ~Concurrency();
    void set_app_concurrency(unsigned int value);
    unsigned int get_app_concurrency();
    unsigned int get_hardware_concurrency();
private:
    Concurrency();
private:
    static boost::mutex _s_mutex;
    static Concurrency* _s_instance;

private:
    boost::mutex _mutex;
    unsigned int _app_concurrency;
    unsigned int _hardware_concurrency;
};

MED_IMAGING_END_NAMESPACE

#endif