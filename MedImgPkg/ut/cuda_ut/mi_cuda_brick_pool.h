//#ifndef MI_CUDA_BRICK_POOL_H
//#define MI_CUDA_BRICK_POOL_H
//
//#include <map>
//#include <memory>
//#include <string>
//
//#include "arithmetic/mi_aabb.h"
//#include "renderalgo/mi_brick_define.h"
//#include "renderalgo/mi_render_algo_export.h"
//#include "mi_cuda_vr_common.h"
//
//MED_IMG_BEGIN_NAMESPACE
//
//// GPU rendering use brick pool to accelerate
//class ImageData;
//class CudaVolumeBrickInfoCalculator;
//class CudaMaskBrickInfoCalculator;
//class CudaBrickPool {
//public:
//    CudaBrickPool(unsigned int brick_size, unsigned int brick_margin);
//    ~CudaBrickPool();
//
//    void set_volume(std::shared_ptr<ImageData> image_data);
//    void set_mask(std::shared_ptr<ImageData> mask_data);
//
//    void set_volume_texture(cudaGLTextureReadOnly tex);
//    void set_mask_texture(cudaGLTextureReadOnly tex);
//
//    unsigned int get_brick_size() const;
//    unsigned int get_brick_margin() const;
//
//    void get_brick_dim(unsigned int(&brick_dim)[3]);
//    unsigned int get_brick_count() const;
//
//    void calculate_brick_geometry();
//    const BrickGeometry& get_brick_geometry() const;
//
//    void calculate_volume_brick_info();
//    VolumeBrickInfo* get_volume_brick_info() const;
//    void write_volume_brick_info(const std::string& path);
//
//    void add_visible_labels_cache(const std::vector<unsigned char>& vis_labels);
//    void get_visible_labels_cache(std::vector<std::vector<unsigned char>>& vis_labels);
//    void clear_visible_labels_cache();
//    std::vector<std::vector<unsigned char>> get_stored_visible_labels();
//
//    void calculate_mask_brick_info(const std::vector<unsigned char>& vis_labels);
//    void update_mask_brick_info(const AABBUI& aabb);
//    MaskBrickInfo* get_mask_brick_info(const std::vector<unsigned char>& vis_labels) const;
//    void write_mask_brick_info(const std::string& path,
//        const std::vector<unsigned char>& visible_labels);
//
//    void remove_mask_brick_info(const std::vector<unsigned char>& vis_labels);
//    void remove_all_mask_brick_info();
//
//public:
//    void calculate_intercect_brick_range(const AABB& bounding,
//        AABBI& brick_range);
//    void get_clipping_brick_geometry(const AABB& bounding, float* brick_vertex,
//        float* brick_color);
//
//public:
//    void debug_save_mask_info(const std::string& path);
//
//private:
//    std::shared_ptr<ImageData> _volume;
//    std::shared_ptr<ImageData> _mask;
//    cudaGLTextureReadOnly _volume_texture;
//    cudaGLTextureReadOnly _mask_texture;
//
//    unsigned int _brick_size;
//    unsigned int _brick_margin;
//    unsigned int _brick_dim[3];
//    unsigned int _brick_count;
//
//    BrickGeometry _brick_geometry; // For GL rendering
//
//    std::unique_ptr<VolumeBrickInfo[]> _h_volume_brick_info_array;
//    float* _d_volume_brick_info_buffer;//CUDA device global memory
//
//    std::map<LabelKey, std::unique_ptr<MaskBrickInfo[]>> _mask_brick_info_array_set;
//    std::map<LabelKey, int* > _mask_brick_info_buffer_set;//CUDA device global memory
//
//    CudaMemShield _cuda_mem_shield;
//
//    std::map<LabelKey, std::vector<unsigned char>> _vis_labels_cache;
//
//private:
//    std::unique_ptr<CudaVolumeBrickInfoCalculator> _volume_brick_info_cal;
//    std::unique_ptr<CudaMaskBrickInfoCalculator> _mask_brick_info_cal;
//
//private:
//    DISALLOW_COPY_AND_ASSIGN(CudaBrickPool);
//};
//
//MED_IMG_END_NAMESPACE
//#endif