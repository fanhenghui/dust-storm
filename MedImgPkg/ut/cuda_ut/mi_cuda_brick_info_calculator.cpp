//#include "mi_cuda_brick_info_calculator.h"
//#include "io/mi_image_data.h"
//
//MED_IMG_BEGIN_NAMESPACE
//
//CudaBrickInfoCalculator::CudaBrickInfoCalculator()
//    : _h_info_array(nullptr), _brick_size(16), _brick_margin(2) {
//    _brick_dim[0] = 1;
//    _brick_dim[1] = 1;
//    _brick_dim[2] = 1;
//}
//
//CudaBrickInfoCalculator::~CudaBrickInfoCalculator() {}
//
//void CudaBrickInfoCalculator::set_brick_margin(unsigned int brick_margin) {
//    _brick_margin = brick_margin;
//}
//
//void CudaBrickInfoCalculator::set_brick_dim(unsigned int(&brick_dim)[3]) {
//    memcpy(_brick_dim, brick_dim, sizeof(unsigned int) * 3);
//}
//
//void CudaBrickInfoCalculator::set_brick_size(unsigned int brick_size) {
//    _brick_size = brick_size;
//}
//
//void CudaBrickInfoCalculator::set_brick_info_array_host(char* info_array) {
//    _h_info_array = info_array;
//}
//
//void CudaBrickInfoCalculator::set_brick_info_array_device(char* info_array) {
//    _d_info_array = info_array;
//}
//
//void CudaBrickInfoCalculator::set_data_texture(cudaGLTextureReadOnly tex) {
//    _img_texture = tex;
//}
//
//void CudaBrickInfoCalculator::set_data(std::shared_ptr<ImageData> img) {
//    _img_data = img;
//}
//
//CudaVolumeBrickInfoCalculator::CudaVolumeBrickInfoCalculator() {}
//
//CudaVolumeBrickInfoCalculator::~CudaVolumeBrickInfoCalculator() {}
//
//void CudaVolumeBrickInfoCalculator::calculate() {
//    initialize();
//    calculate_gpu();
//    download();
//}
//
//void CudaVolumeBrickInfoCalculator::download() {
//    
//}
//
//void CudaVolumeBrickInfoCalculator::calculate_gpu() {
//    
//}
//
//void CudaVolumeBrickInfoCalculator::initialize() {
//    
//}
//
//CudaMaskBrickInfoCalculator::CudaMaskBrickInfoCalculator() {}
//
//CudaMaskBrickInfoCalculator::~CudaMaskBrickInfoCalculator() {}
//
//void CudaMaskBrickInfoCalculator::update(const AABBUI& aabb) {
//    initialize();
//    update_gpu(aabb);
//    download();
//}
//
//void CudaMaskBrickInfoCalculator::calculate() {
//    initialize();
//    calculate_gpu();
//    download();
//}
//
//void CudaMaskBrickInfoCalculator::download() {
//    
//}
//
//void CudaMaskBrickInfoCalculator::calculate_gpu() {
//    AABBUI aabb;
//    aabb._min[0] = 0;
//    aabb._min[1] = 0;
//    aabb._min[2] = 0;
//
//    aabb._max[0] = _img_data->_dim[0];
//    aabb._max[1] = _img_data->_dim[1];
//    aabb._max[2] = _img_data->_dim[2];
//
//    this->update_gpu(aabb);
//}
//
//void CudaMaskBrickInfoCalculator::update_gpu(const AABBUI& aabb) {
//    
//}
//
//void CudaMaskBrickInfoCalculator::initialize() {
//    
//}
//
//void CudaMaskBrickInfoCalculator::set_visible_labels(
//    const std::vector<unsigned char>& labels) {
//    _visible_labels = labels;
//}
//
//MED_IMG_END_NAMESPACE
