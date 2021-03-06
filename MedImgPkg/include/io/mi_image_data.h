#ifndef MEDIMGIO_IMAGE_DATA_H
#define MEDIMGIO_IMAGE_DATA_H

#include "arithmetic/mi_matrix4.h"
#include "arithmetic/mi_point3.h"
#include "io/mi_io_define.h"
#include "io/mi_io_export.h"
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class IO_Export ImageData {
public:
    ImageData();

    ~ImageData();

    bool mem_allocate();

    float get_min_scalar();

    float get_max_scalar();

    //regulate_to_positive flag is for GPU texture
    bool regulate_wl(float& window, float& level, bool regulate_to_positive = true);

    void normalize_wl(float& window, float& level);

    bool regulate_pixel_value(float& pixel, bool regulate_to_positive = true);

    void normalize_pixel_value(float& pixel);

    void get_pixel_value(unsigned int x, unsigned int y, unsigned int z,
                         double& value) const;

    void get_pixel_value(const Point3& pos, double& value) const;

    void set_data_dirty();

    void* get_pixel_pointer();

    unsigned int get_data_size();

    void shallow_copy(ImageData* image_data);

    void deep_copy(ImageData* image_data);

public:
    DataType _data_type;
    unsigned int _channel_num;
    unsigned int _dim[3];

    double _spacing[3];

    float _slope;
    float _intercept;

    Vector3 _image_orientation[3];
    Point3 _image_position;

private:
    std::unique_ptr<char[]> _data_array;

private:
    template <typename T> void find_min_max(T* data_array);

    void find_min_max();

    float _min_scalar;
    float _max_scalar;
    bool _has_cal_min_max;
};

template <typename T> void ImageData::find_min_max(T* data_array) {
    if (nullptr == data_array) {
        return;
    }

    T min = *data_array, max = *data_array;
    unsigned int size = _dim[0] * _dim[1] * _dim[2] * _channel_num;

    for (unsigned int i = 1; i < size; i++) {
        T temp = *(data_array + i);
        min = temp < min ? temp : min;
        max = temp > max ? temp : max;
    }

    _min_scalar = static_cast<float>(min);
    _max_scalar = static_cast<float>(max);
}

MED_IMG_END_NAMESPACE

#endif