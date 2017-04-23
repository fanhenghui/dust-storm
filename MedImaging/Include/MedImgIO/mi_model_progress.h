#ifndef MED_IMAGING_MODEL_PROGRESS
#define MED_IMAGING_MODEL_PROGRESS

#include "MedImgIO/mi_io_stdafx.h"
#include "MedImgCommon/mi_model_interface.h"

MED_IMAGING_BEGIN_NAMESPACE

class IO_Export ProgressModel : public IModel
{
public:
    ProgressModel():_progress(0)
    {}

    virtual ~ProgressModel() {};

    void set_progress(int value) 
    {
        _progress = value;
        set_changed();
    }

    int get_progress() const
    {
        return _progress;
    }

protected:
private:
    int _progress;
};

MED_IMAGING_END_NAMESPACE

#endif