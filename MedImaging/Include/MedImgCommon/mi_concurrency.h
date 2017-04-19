#ifndef MED_IMAGING_CONCURRENCY_H_
#define MED_IMAGING_CONCURRENCY_H_

#include "MedImgCommon/mi_common_stdafx.h"
#include "boost/thread/mutex.hpp"

MED_IMAGING_BEGIN_NAMESPACE

class Common_Export Concurrency
{
public:
    static Concurrency* Instance();
    ~Concurrency();
    void SetAppConcurrency(unsigned int uiValue);
    unsigned int GetAppConcurrency();
    unsigned int GetHardwareConcurrency();
private:
    Concurrency();
private:
    static boost::mutex m_mutexStatic;
    static Concurrency* m_instance;

private:
    boost::mutex m_mutex;
    unsigned int m_uiAppConcurrency;
    unsigned int m_uiHardwareConcurrency;
};

MED_IMAGING_END_NAMESPACE

#endif