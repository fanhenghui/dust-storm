#include "mi_cuda_texture_1d_array.h"
#include "mi_cuda_texture_1d.h"
#include "mi_cuda_utils.h"
#include "mi_cuda_resource_logger.h"

MED_IMG_BEGIN_NAMESPACE

CudaTexture1DArray::CudaTexture1DArray(UIDType uid, int array_length) :
    CudaObject(uid, "CudaTexture1DArray"), _length(0), _format(cudaChannelFormatKindNone), _array_length(array_length) {
    _channel[0] = 0;
    _channel[1] = 0;
    _channel[2] = 0;
    _channel[3] = 0;
}

CudaTexture1DArray::~CudaTexture1DArray() {
    finalize();
}

void CudaTexture1DArray::initialize() {

}

void CudaTexture1DArray::finalize() {
    cudaError_t err = cudaSuccess;
    for (auto it = _tex_objs.begin(); it != _tex_objs.end(); ++it) {
        for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
            err = cudaDestroyTextureObject(it2->second);
            it2->second = 0;
            CHECK_CUDA_ERROR(err);
        }
    }
    _tex_objs.clear();

    for (auto it = _cuda_arrays.begin(); it != _cuda_arrays.end(); ++it) {
        err = cudaFreeArray(it->second);
        it->second = nullptr;
        CHECK_CUDA_ERROR(err);
    }
    _cuda_arrays.clear();

    _length = 0;
    _channel[0] = 0;
    _channel[1] = 0;
    _channel[2] = 0;
    _channel[3] = 0;
    _format = cudaChannelFormatKindNone;
    _array_length =0;
}

float CudaTexture1DArray::memory_used() const {
    return 0;
}

int CudaTexture1DArray::load(int channel_x, int channel_y, int channel_z, int channel_w, 
    int position, cudaChannelFormatKind format, int length, void* data) {
    if (length <= 0) {
        MI_CUDARESOURCE_LOG(MI_ERROR) << "load invalid length " << length << " to texture 1D array.";
        return -1;
    }
    if (position > _array_length - 1) {
        MI_CUDARESOURCE_LOG(MI_ERROR) << "load invalid position " << position << " to texture 1D array.";
        return -1;
    }

    //malloc and load, or update all
    cudaArray_t cuda_array = nullptr;
    auto it = _cuda_arrays.find(position);
    if (it != _cuda_arrays.end()) {
        if (_channel[0] != channel_x || _channel[1] != channel_y ||
            _channel[2] != channel_z || _channel[3] != channel_w || length != _length) {
            MI_CUDARESOURCE_LOG(MI_ERROR) << "load different format array to CUDA texture 1D array. init foramt {ch:"
                << _channel[0] << " " << _channel[1] << " " << _channel[2] << " " << _channel[3] << ", format: " << _format << ", length: " << _length
                << "}. call load func foramt {ch: "
                << channel_x << " " << channel_y << " " << channel_z << " " << channel_w << ", format: " << format << ", length: " << length << "}.";
            return -1;
        }
        cuda_array = it->second;
    } else {
        _channel[0] = channel_x;
        _channel[1] = channel_y;
        _channel[2] = channel_z;
        _channel[3] = channel_w;
        _format = format;
        _length = length;
        cudaChannelFormatDesc format_desc = cudaCreateChannelDesc(channel_x, channel_y, channel_z, channel_w, format);
        cudaError_t err = cudaMallocArray(&cuda_array, &format_desc, length, 1);
        if (err != cudaSuccess) {
            LOG_CUDA_ERROR(err);
            return -1;
        }
        _cuda_arrays[position] = cuda_array;
    }

    if (nullptr == data) {
        return 0;
    }

    cudaError_t err = cudaMemcpyToArray(cuda_array, 0, 0, data, length*CudaUtils::get_component_byte(_channel), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        LOG_CUDA_ERROR(err);
        return -1;
    }
    else {
        return 0;
    }
}

void CudaTexture1DArray::get_texture_array(cudaTextureAddressMode address_mode, cudaTextureFilterMode filter_mode,
    cudaTextureReadMode read_mode, bool normalized_coords, cudaTextureObject_t* tex_array) {
    //cudaTextureAddressMode : bit 7~ ...
    //cudaTextureFilterMode : bit 4~6
    //cudaTextureReadMode : bit 1~3
    //normalized Coordinate : bit 0
    int id = 0;
    id |= (int(address_mode)) << 7;
    id |= (int(filter_mode)) << 4;
    id |= (int(read_mode)) << 1;
    id |= (int(normalized_coords));

    for (int i=0; i< _array_length; ++i) {
        auto it_cuda_array = _cuda_arrays.find(i);
        if (it_cuda_array == _cuda_arrays.end()) {
            tex_array[i] = 0;
        } else {
            auto it = _tex_objs.find(i);
            if (it == _tex_objs.end()) {
                cudaTextureObject_t obj_new = create_object(address_mode, filter_mode, read_mode, normalized_coords, it_cuda_array->second);
                std::map<int, cudaTextureObject_t> new_texs;
                new_texs.insert(std::make_pair(id, obj_new));
                _tex_objs.insert(std::make_pair(i, new_texs));
                tex_array[i] = obj_new;
            } else {
                auto it2 = it->second.find(id);
                if (it2 == it->second.end()) {
                    cudaTextureObject_t obj_new = create_object(address_mode, filter_mode, read_mode, normalized_coords, it_cuda_array->second);
                    it->second.insert(std::make_pair(id, obj_new));
                    tex_array[i] = obj_new;
                } else {
                    tex_array[i] = it2->second;
                }
            }
        }
    }
}

cudaTextureObject_t CudaTexture1DArray::create_object(
    cudaTextureAddressMode address_mode, cudaTextureFilterMode filter_mode,
    cudaTextureReadMode read_mode, bool normalized_coords, cudaArray_t cuda_array) {

    struct cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(cudaResourceDesc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cuda_array;

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(cudaTextureDesc));
    tex_desc.addressMode[0] = address_mode;
    tex_desc.filterMode = filter_mode;
    tex_desc.readMode = read_mode;
    tex_desc.normalizedCoords = normalized_coords;

    cudaTextureObject_t tex_obj(0);
    cudaError_t err = cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, NULL);
    CHECK_CUDA_ERROR(err);

    return tex_obj;
}

int CudaTexture1DArray::get_array_length() const {
    return _array_length;
}

int CudaTexture1DArray::get_length() const {
    return _length;
}

MED_IMG_END_NAMESPACE


