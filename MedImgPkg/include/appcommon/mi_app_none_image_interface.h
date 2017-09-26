#ifndef MED_IMH_APP_NONE_IMAGE_H
#define MED_IMH_APP_NONE_IMAGE_H

class IAppNoneImage {
public:
    IAppNoneImage():_dirty(false) {};
    virtual ~IAppNoneImage() {};

    void set_dirty(bool flag) {_dirty = flag;};
    bool get_dirty() const {return _dirty;};

    virtual bool check_dirty() {
        if(get_dirty()) {
            return true;
        } else {
            return check_dirty_i();
        }
    }

    virtual void update() = 0;
    virtual char* serialize_dirty(int buffer_size) const = 0;
protected:
    virtual bool check_dirty_i();
private:
    bool _dirty;
};

#endif