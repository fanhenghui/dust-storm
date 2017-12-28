//#ifndef MI_CUDA_BRICK_INFO_CALCULATOR_H
//#define MI_CUDA_BRICK_INFO_CALCULATOR_H
//
//#include "arithmetic/mi_aabb.h"
//#include "renderalgo/mi_brick_define.h"
//#include "renderalgo/mi_render_algo_export.h"
//#include <memory>
//#include "mi_cuda_vr_common.h"
//
//MED_IMG_BEGIN_NAMESPACE
//
//class ImageData;
//class CudaBrickPool;
//
//class CudaBrickInfoCalculator {
//public:
//    CudaBrickInfoCalculator();
//    virtual ~CudaBrickInfoCalculator();
//
//    void set_data(std::shared_ptr<ImageData> img);
//    void set_data_texture(cudaGLTextureReadOnly tex);
//    void set_brick_info_array_device(char* d_info_array);
//    void set_brick_info_array_host(char* h_info_array);
//
//    void set_brick_size(unsigned int brick_size);
//    void set_brick_dim(unsigned int(&brick_dim)[3]);
//    void set_brick_margin(unsigned int brick_margin);
//
//    virtual void calculate() = 0;
//
//protected:
//    std::shared_ptr<ImageData> _img_data;
//    cudaGLTextureReadOnly _img_texture;
//    char* _d_info_array;
//    char* _h_info_array;
//
//    unsigned int _brick_size;
//    unsigned int _brick_dim[3];
//    unsigned int _brick_margin;
//
//private:
//    DISALLOW_COPY_AND_ASSIGN(CudaBrickInfoCalculator);
//};
//
//class CudaVolumeBrickInfoCalculator : public CudaBrickInfoCalculator {
//public:
//    CudaVolumeBrickInfoCalculator();
//    virtual ~CudaVolumeBrickInfoCalculator();
//
//    virtual void calculate();
//
//private:
//    void initialize();
//    void calculate_gpu();
//    void download();
//};
//
//class CudaMaskBrickInfoCalculator : public CudaBrickInfoCalculator {
//public:
//    CudaMaskBrickInfoCalculator();
//    virtual ~CudaMaskBrickInfoCalculator();
//
//    void set_visible_labels(const std::vector<unsigned char>& labels);
//
//    virtual void calculate();
//    void update(const AABBUI& aabb);
//
//private:
//    void initialize();
//    void calculate_gpu();
//    void update_gpu(const AABBUI& aabb);
//    void download();
//
//private:
//    int *_d_visible_labels;
//    std::vector<unsigned char> _visible_labels;
//};
//
//MED_IMG_END_NAMESPACE
//
//#endif