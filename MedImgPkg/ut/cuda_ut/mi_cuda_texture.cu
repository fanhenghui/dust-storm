#include "GL/glew.h"
#include "GL/freeglut.h"

#include <cuda_runtime.h>
#include <cuda.h>  
#include <cuda_gl_interop.h>
#include <cuda_texture_types.h>
#include <vector_types.h>
#include <iostream>
#include <fstream>


#define CHECK_CUDA_ERROR {\
cudaError_t err = cudaGetLastError(); \
if (err != cudaSuccess) {\
    std::cout << "CUDA error: " << err << " in function: " << __FUNCTION__ <<\
    " line: " << __LINE__ << std::endl; \
}}\

#define BLOCK_SIZE 16

static int _width = 1024;
static int _height = 1024;
static unsigned char* _checkboard_data = nullptr;
static unsigned char* _checkboard_data_host = nullptr;
static unsigned char* _output = nullptr;
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> tex_reference;

void create_data() {
    _checkboard_data = new unsigned char[_width*_height * 4];
    int tag_x = 0;
    int tag_y = 0;
    int idx = 0;
    for (int y = 0; y < _height; ++y) {
        for (int x = 0; x < _width; ++x) {
            tag_x = x / 32;
            tag_y = y / 32;
            idx = y*_width + x;
            if ((tag_x + tag_y) % 2 == 0) {
                _checkboard_data[idx * 4] = 200;
                _checkboard_data[idx * 4 + 1] = 200;
                _checkboard_data[idx * 4 + 2] = 200;
                _checkboard_data[idx * 4 + 3] = 255;
            }
            else {
                _checkboard_data[idx * 4] = 20;
                _checkboard_data[idx * 4 + 1] = 20;
                _checkboard_data[idx * 4 + 2] = 20;
                _checkboard_data[idx * 4 + 3] = 255;
            }
        }
    }
}

__global__ void tansfromKernelTexRef(unsigned char* output, int width, int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    float u = x / (float)width;
    float v = y / (float)height;

    uchar4 rgba = tex2D(tex_reference, x, y);
    int idx = y*width + x;
    output[idx * 4] = (unsigned char)(rgba.x);
    output[idx * 4 + 1] = 20;//  change R 
    output[idx * 4 + 2] = (unsigned char)(rgba.z);
    output[idx * 4 + 3] = 255;
}

__global__ void tansfromKernelTexObj(unsigned char* output, cudaTextureObject_t tex_obj, int width, int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    float u = x / (float)width;
    float v = y / (float)height;

    uchar4 rgba = tex2D<uchar4>(tex_obj, x, y);
    int idx = y*width + x;
    output[idx * 4] = (unsigned char)(rgba.x);
    output[idx * 4 + 1] = 20;//  change R 
    output[idx * 4 + 2] = (unsigned char)(rgba.z);
    output[idx * 4 + 3] = 255;
}

__global__ void tansfromKernelTexObjFloat(unsigned char* output, cudaTextureObject_t tex_obj, int width, int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    float u = x / (float)width;
    float v = y / (float)height;

    float4 rgba = tex2D<float4>(tex_obj, u, v);
    int idx = y*width + x;
    output[idx * 4] = (unsigned char)(rgba.x*255);
    output[idx * 4 + 1] = 20;//  change R 
    output[idx * 4 + 2] = (unsigned char)(rgba.z*255);
    output[idx * 4 + 3] = 255;
}


//Obj API
void cuda_texture_obj() {
    //CUDA array
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(
        8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray* cuda_array;
    cudaMallocArray(&cuda_array, &channel_desc, _width, _height);

    CHECK_CUDA_ERROR;

    //copy data to CUDA array
    cudaMemcpyToArray(cuda_array, 0, 0, _checkboard_data, _width*_height * 4, cudaMemcpyHostToDevice);

    CHECK_CUDA_ERROR;

    //Cuda resource
    struct cudaResourceDesc  res_desc;
    memset(&res_desc, 0, sizeof(cudaResourceDesc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cuda_array;
    
    //Texture parameter (like GL's glTexParameteri)
    struct cudaTextureDesc tex_desc;
    memset(&tex_desc,0, sizeof(cudaTextureDesc));
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModePoint;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 0;

    //create texture
    cudaTextureObject_t tex_obj = 0;
    cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, NULL);

    CHECK_CUDA_ERROR;

    //invoke
#define BLOCK_SIZE 16
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(_width / BLOCK_SIZE, _height / BLOCK_SIZE);
    tansfromKernelTexObj << <grid, block >> >( _output, tex_obj, _width, _height);
}

//Obj API
void cuda_texture_obj_float() {
    //CUDA array
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(
        8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray* cuda_array;
    cudaMallocArray(&cuda_array, &channel_desc, _width, _height);

    CHECK_CUDA_ERROR;

    //copy data to CUDA array
    cudaMemcpyToArray(cuda_array, 0, 0, _checkboard_data, _width*_height * 4, cudaMemcpyHostToDevice);

    CHECK_CUDA_ERROR;

    //Cuda resource
    struct cudaResourceDesc  res_desc;
    memset(&res_desc, 0, sizeof(cudaResourceDesc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cuda_array;

    //Texture parameter (like GL's glTexParameteri)
    struct cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(cudaTextureDesc));
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 1;

    //create texture
    cudaTextureObject_t tex_obj = 0;
    cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, NULL);

    CHECK_CUDA_ERROR;

    //invoke
#define BLOCK_SIZE 16
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(_width / BLOCK_SIZE, _height / BLOCK_SIZE);
    tansfromKernelTexObjFloat << <grid, block >> >(_output, tex_obj, _width, _height);
}

void cuda_texture_ref_cuda_array() {
    //use CUDA array
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(
        8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray* cuda_array;
    cudaMallocArray(&cuda_array, &channel_desc, _width, _height);

    CHECK_CUDA_ERROR;

    //copy data to CUDA array
    cudaMemcpyToArray(cuda_array, 0, 0, _checkboard_data, _width*_height * 4, cudaMemcpyHostToDevice);

    cudaBindTextureToArray(&tex_reference, cuda_array, &channel_desc);

    CHECK_CUDA_ERROR;

    unsigned char* output = nullptr;
    

    CHECK_CUDA_ERROR;

    //invoke
#define BLOCK_SIZE 16
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(_width / BLOCK_SIZE, _height / BLOCK_SIZE);
    tansfromKernelTexRef << <grid, block >> >(_output, _width, _height);

    cudaUnbindTexture(&tex_reference);
}

extern "C"  
int _cuda_texture(int argc, char* argv[]) {
    create_data();  
    
    //create output
    cudaMalloc(&_output, _width*_height * sizeof(unsigned char) * 4);
    CHECK_CUDA_ERROR;

    //calculate based on texture
    cuda_texture_obj();
    CHECK_CUDA_ERROR;

    //device memory to host
    _checkboard_data_host = new unsigned char[_width*_height*4];
    cudaMemcpy(_checkboard_data_host, _output, _width*_height*4, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR; 

    std::ofstream out("D:/temp/tex_res.raw", std::ios::out | std::ios::binary );
    if (out.is_open()) {
        out.write((char*)_checkboard_data_host, _width*_height*4);
        out.close();
        std::cout << "write done.";
    }
    std::cout << "done.";

    return 0;
}

