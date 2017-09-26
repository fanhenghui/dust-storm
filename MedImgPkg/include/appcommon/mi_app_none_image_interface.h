#ifndef MED_IMH_APP_NONE_IMAGE_INTERFACE_H
#define MED_IMH_APP_NONE_IMAGE_INTERFACE_H

#include "appcommon/mi_app_common_export.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export IAppNoneImage {
public:
    IAppNoneImage():_dirty(false) {};
    virtual ~IAppNoneImage() {};

    void set_dirty(bool flag) {_dirty = flag;};
    bool get_dirty() const {return _dirty;};

    bool check_dirty() {
        if(get_dirty()) {
            return true;
        } else {
            return check_dirty_i();
        }
    }

    virtual void update() = 0;
    virtual char* serialize_dirty(int buffer_size) const = 0;
protected:
    virtual bool check_dirty_i() = 0;
private:
    bool _dirty;
};

MED_IMG_END_NAMESPACE
#endif