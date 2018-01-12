#ifndef MED_IMG_CUDARESOUECE_MI_CUDA_TIME_QUERY_H
#define MED_IMG_CUDARESOUECE_MI_CUDA_TIME_QUERY_H

#include <cuda_runtime.h>
#include "cudaresource/mi_cuda_object.h"

MED_IMG_BEGIN_NAMESPACE

class CUDAResource_Export CudaTimeQuery : public CudaObject
{
public:
    CudaTimeQuery(UIDType uid);
    virtual ~CudaTimeQuery();

    virtual void initialize();
    virtual void finalize();
    virtual float memory_used() const;

    void begin();
    float end();
    float get_time_elapsed() const;

private:
    cudaEvent_t _start;
    cudaEvent_t _end;
    float _time_elapsed;
};

class CUDAResource_Export ScopedCudaTimeQuery
{
public:
    ScopedCudaTimeQuery(std::shared_ptr<CudaTimeQuery> tq, float* recorder) :_time_query(tq), _recorder(recorder) {
        if (_time_query) {
            _time_query->begin();
        }
    }

    ~ScopedCudaTimeQuery() {
        if (_time_query) {
            if (_recorder) {
                *_recorder = _time_query->end();
            }
        }
    }
private:
    std::shared_ptr<CudaTimeQuery> _time_query;
    float* _recorder;

private:
    DISALLOW_COPY_AND_ASSIGN(ScopedCudaTimeQuery);
};

MED_IMG_END_NAMESPACE
#endif
