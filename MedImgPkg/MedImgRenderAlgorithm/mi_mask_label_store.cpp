#include "mi_mask_label_store.h"

MED_IMG_BEGIN_NAMESPACE


MaskLabelStore* MaskLabelStore::instance()
{
    if (nullptr == _S_INSTANCE)
    {
        boost::unique_lock<boost::mutex> locker(_S_MUTEX);
        if (nullptr == _S_INSTANCE)
        {
            _S_INSTANCE = new MaskLabelStore();
        }
    }

    return _S_INSTANCE;
}

MaskLabelStore::~MaskLabelStore()
{

}

unsigned char MaskLabelStore::acquire_label()
{
    boost::unique_lock<boost::mutex> locker(_mutex);

    unsigned char candidate(0); 
    for (unsigned char i = 1 ; i < 255 ; ++i)
    {
        if (0 == _label_store[i])
        {
            _label_store[i] = 1;
            candidate = i;
            break;
        }
    }
    if (0 == candidate)
    {
        RENDERALGO_THROW_EXCEPTION("mask store is empty!");
    }

    return candidate;
}

std::vector<unsigned char> MaskLabelStore::acquire_labels(int num)
{
    boost::unique_lock<boost::mutex> locker(_mutex);

    std::vector<unsigned char> candidates(num);
    int idx = 0;
    for (unsigned char i = 1 ; i < 255 ; ++i)
    {
        if (0 == _label_store[i])
        {
            _label_store[i] = 1;
            candidates[idx++] = i;
            if (idx >= num)
            {
                break;
            }
        }
    }
    if (idx < num)
    {
        RENDERALGO_THROW_EXCEPTION("mask store is empty!");
    }

    return candidates;
}

void MaskLabelStore::recycle_label(unsigned char label)
{
    boost::unique_lock<boost::mutex> locker(_mutex);

    _label_store[label] = 0;
}

void MaskLabelStore::recycle_labels(std::vector<unsigned char> labels)
{
    boost::unique_lock<boost::mutex> locker(_mutex);

    for (auto it = labels.begin() ; it != labels.end() ; ++it)
    {
        _label_store[*it] = 0;
    }
}

MaskLabelStore::MaskLabelStore()
{
    memset(_label_store , 0 , 255);
}

boost::mutex MaskLabelStore::_S_MUTEX;

MaskLabelStore* MaskLabelStore::_S_INSTANCE = nullptr;

MED_IMG_END_NAMESPACE