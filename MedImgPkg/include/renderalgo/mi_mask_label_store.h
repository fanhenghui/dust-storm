#ifndef MEDIMGRENDERALGO_LABEL_STORE_H
#define MEDIMGRENDERALGO_LABEL_STORE_H

#include "boost/thread/locks.hpp"
#include "boost/thread/mutex.hpp"
#include "renderalgo/mi_render_algo_export.h"
#include <vector>

MED_IMG_BEGIN_NAMESPACE

class RenderAlgo_Export MaskLabelStore {
public:
    static MaskLabelStore* instance();
    ~MaskLabelStore();

    void fill_label(unsigned char label);
    void fill_labels(std::vector<unsigned char> labels);

    unsigned char acquire_label();
    std::vector<unsigned char> acquire_labels(int num);

    void recycle_label(unsigned char label);
    void recycle_labels(std::vector<unsigned char> labels);

    void reset_labels();

private:
    unsigned char _label_store[255];
    boost::mutex _mutex;

private:
    MaskLabelStore();

    static MaskLabelStore* _S_INSTANCE;
    static boost::mutex _S_MUTEX;
};

MED_IMG_END_NAMESPACE

#endif