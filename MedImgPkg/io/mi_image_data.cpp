#include "mi_image_data.h"
#include "arithmetic/mi_sampler.h"
#include "util/mi_configuration.h"

MED_IMG_BEGIN_NAMESPACE

ImageData::ImageData()
    : _data_type(SHORT), _channel_num(1), _min_scalar(0), _max_scalar(1024),
      _slope(1.0f), _intercept(0.0f), _image_position(Point3::S_ZERO_POINT),
      _has_cal_min_max(false) {
  memset(_dim, 0, sizeof(_dim));
  memset(_dim, 0, sizeof(_spacing));

  _image_position = Point3(0.0, 0.0, 0.0);

  _image_orientation[0] = Vector3(1.0, 0.0, 0.0);
  _image_orientation[1] = Vector3(0.0, 1.0, 0.0);
  _image_orientation[2] = Vector3(0.0, 0.0, 1.0);
}

ImageData::~ImageData() {}

bool ImageData::mem_allocate() {
  const unsigned int mem_size = get_data_size();
  _data_array.reset(new char[mem_size]);
  memset(_data_array.get(), 0, mem_size);
  _has_cal_min_max = false;
  return true;
}

float ImageData::get_min_scalar() {
  if (!_has_cal_min_max) {
    find_min_max_i();
  }
  return _min_scalar;
}

float ImageData::get_max_scalar() {
  if (!_has_cal_min_max) {
    find_min_max_i();
  }
  return _max_scalar;
}

bool ImageData::regulate_wl(float &window, float &level) {
  // CT should apply slope and intercept
  // MR has always slope(1) and intercept(0)
  if (_slope < DOUBLE_EPSILON) {
    return false;
  }

  if (Configuration::instance()->get_processing_unit_type() == GPU) {
    const float min_gray = get_min_scalar();
    if (_data_type == SHORT || _data_type == CHAR) {
      level = (level - _intercept - min_gray) / _slope;
    } else {
      level = (level - _intercept) / _slope;
    }
  } else {
    level = (level - _intercept) / _slope;
  }

  window = window / _slope;

  return true;
}

void ImageData::normalize_wl(float &window, float &level) {
  const static float S_65535_R = 1.0f / 65535.0f;
  const static float S_255_R = 1.0f / 255.0f;

  switch (_data_type) {
  case USHORT:
    window *= S_65535_R;
    level *= S_65535_R;
    break;
  case SHORT:
    window *= S_65535_R;
    level = level * S_65535_R;
    break;
  case UCHAR:
    window *= S_255_R;
    level *= S_255_R;
    break;
  case CHAR:
    window *= S_255_R;
    level = level * S_255_R;
    break;
  case FLOAT:
    break;
  default:
    IO_THROW_EXCEPTION("Undefined image type!");
  }
}

bool ImageData::regulate_normalize_wl(float &window, float &level) {
  if (!regulate_wl(window, level)) {
    return false;
  }

  normalize_wl(window, level);

  return true;
}

void ImageData::get_pixel_value(unsigned int x, unsigned int y, unsigned int z,
                                double &value) const {
  void *data_array = _data_array.get();
  if (nullptr == data_array) {
    IO_THROW_EXCEPTION("Volume data is null!");
  }

  if (x >= _dim[0] || y >= _dim[1] || z >= _dim[2]) {
    IO_THROW_EXCEPTION("Input coordinate is out of data range!");
  }

  double pixel_value = 0.0;
  unsigned int offset = x + y * _dim[0] + z * _dim[0] * _dim[1];
  switch (_data_type) {
  case CHAR: {
    char *char_array = (char *)data_array;
    pixel_value = static_cast<double>(char_array[offset]);
    break;
  }
  case UCHAR: {
    unsigned char *unsigned_char_array = (unsigned char *)data_array;
    pixel_value = static_cast<double>(unsigned_char_array[offset]);
    break;
  }
  case USHORT: {
    unsigned short *unsigned_short_array = (unsigned short *)data_array;
    pixel_value = static_cast<double>(unsigned_short_array[offset]);
    break;
  }
  case SHORT: {
    short *short_array = (short *)data_array;
    pixel_value = static_cast<double>(short_array[offset]);
    break;
  }
  case FLOAT: {
    float *float_array = (float *)data_array;
    pixel_value = static_cast<double>(float_array[offset]);
    break;
  }
  default:
    IO_THROW_EXCEPTION("Undefined image type!");
  }

  value = pixel_value;
}

void ImageData::get_pixel_value(const Point3 &pos, double &value) const {
  void *data_array = _data_array.get();
  if (nullptr == data_array) {
    IO_THROW_EXCEPTION("Undefined image type!");
  }

  if (pos.x > _dim[0] - 1 || pos.y > _dim[1] - 1 || pos.z > _dim[2] - 1) {
    IO_THROW_EXCEPTION("Input coordinate is out of data range!");
  }

  // Find the pixel value
  double pixel_value = 0.0;
  switch (_data_type) {
  case CHAR: {
    Sampler<char> sampler;
    pixel_value = (double)sampler.sample_3d_linear(
        (float)pos.x, (float)pos.y, (float)pos.z, _dim[0], _dim[1], _dim[2],
        (char *)data_array);
    break;
  }
  case UCHAR: {
    Sampler<unsigned char> sampler;
    pixel_value = (double)sampler.sample_3d_linear(
        (float)pos.x, (float)pos.y, (float)pos.z, _dim[0], _dim[1], _dim[2],
        (unsigned char *)data_array);
    break;
  }
  case USHORT: {
    Sampler<unsigned short> sampler;
    pixel_value = (double)sampler.sample_3d_linear(
        (float)pos.x, (float)pos.y, (float)pos.z, _dim[0], _dim[1], _dim[2],
        (unsigned short *)data_array);
    break;
  }
  case SHORT: {
    Sampler<short> sampler;
    pixel_value = (double)sampler.sample_3d_linear(
        (float)pos.x, (float)pos.y, (float)pos.z, _dim[0], _dim[1], _dim[2],
        (short *)data_array);
    break;
  }
  case FLOAT: {
    Sampler<float> sampler;
    pixel_value = (float)sampler.sample_3d_linear(
        (float)pos.x, (float)pos.y, (float)pos.z, _dim[0], _dim[1], _dim[2],
        (float *)data_array);
    break;
  }
  default:
    IO_THROW_EXCEPTION("Undefined image type!");
  }
  value = pixel_value;
}

void ImageData::set_data_dirty() { _has_cal_min_max = false; }

void *ImageData::get_pixel_pointer() { return _data_array.get(); }

void ImageData::shallow_copy(ImageData *image_data) {
  IO_CHECK_NULL_EXCEPTION(image_data);

#define COPY_PARAMETER(p) image_data->p = p
  COPY_PARAMETER(_data_type);
  COPY_PARAMETER(_channel_num);
  COPY_PARAMETER(_slope);
  COPY_PARAMETER(_intercept);
  COPY_PARAMETER(_image_orientation[0]);
  COPY_PARAMETER(_image_orientation[1]);
  COPY_PARAMETER(_image_orientation[2]);
  COPY_PARAMETER(_image_position);
  COPY_PARAMETER(_dim[0]);
  COPY_PARAMETER(_dim[1]);
  COPY_PARAMETER(_dim[2]);
  COPY_PARAMETER(_spacing[0]);
  COPY_PARAMETER(_spacing[1]);
  COPY_PARAMETER(_spacing[2]);
  COPY_PARAMETER(_min_scalar);
  COPY_PARAMETER(_max_scalar);
  COPY_PARAMETER(_has_cal_min_max);
#undef COPY_PARAMETER
}

void ImageData::deep_copy(ImageData *image_data) {
  this->shallow_copy(image_data);

  // Copy this image data
  image_data->mem_allocate();
  const size_t imemSize = this->get_data_size();
  memcpy(image_data->_data_array.get(), this->_data_array.get(), imemSize);
}

void ImageData::find_min_max_i() {
  void *data_array = _data_array.get();
  if (nullptr == data_array) {
    IO_THROW_EXCEPTION("Volume data is null!");
  }

  switch (_data_type) {
  case CHAR:
    this->find_min_max_i((char *)data_array);
    break;
  case UCHAR:
    this->find_min_max_i((unsigned char *)data_array);
    break;
  case USHORT:
    this->find_min_max_i((unsigned short *)data_array);
    break;
  case SHORT:
    this->find_min_max_i((short *)data_array);
    break;
  case FLOAT:
    this->find_min_max_i((float *)data_array);
    break;
  default:
    IO_THROW_EXCEPTION("Undefined image type!");
  }

  _has_cal_min_max = true;
}

unsigned int ImageData::get_data_size() {
  unsigned int imemSize = _dim[0] * _dim[1] * _dim[2] * _channel_num;
  switch (_data_type) {
  case CHAR:
    imemSize *= sizeof(char);
    break;
  case UCHAR:
    imemSize *= sizeof(unsigned char);
    break;
  case USHORT:
    imemSize *= sizeof(unsigned short);
    break;
  case SHORT:
    imemSize *= sizeof(short);
    break;
  case FLOAT:
    imemSize *= sizeof(float);
    break;
  default:
    IO_THROW_EXCEPTION("Undefined image type!");
  }

  return imemSize;
}

MED_IMG_END_NAMESPACE