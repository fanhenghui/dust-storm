#ifndef MED_IMG_MODEL_PROGRESS
#define MED_IMG_MODEL_PROGRESS

#include "MedImgIO/mi_io_export.h"
#include "MedImgUtil/mi_model_interface.h"

MED_IMG_BEGIN_NAMESPACE

class IO_Export ProgressModel : public IModel
{
public:
    ProgressModel():_progress(0)
    {}

    virtual ~ProgressModel() {};

    void set_progress(int value) 
    {
        if (value != _progress)
        {
            _progress = value;
            set_changed();
        }
    }

    int get_progress() const
    {
        return _progress;
    }

protected:
private:
    int _progress;
};

MED_IMG_END_NAMESPACE

#endif