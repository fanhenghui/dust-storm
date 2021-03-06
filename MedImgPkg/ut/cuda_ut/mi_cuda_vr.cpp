#include "GL/glew.h"

#include "io/mi_configure.h"
#include "util/mi_file_util.h"
#include "log/mi_logger.h"

#include "io/mi_dicom_loader.h"
#include "io/mi_image_data.h"
#include "io/mi_image_data_header.h"
#include "io/mi_io_define.h"

#include "arithmetic/mi_ortho_camera.h"
#include "arithmetic/mi_run_length_operator.h"
#include "arithmetic/mi_vector2f.h"

#include "glresource/mi_gl_environment.h"
#include "glresource/mi_gl_fbo.h"
#include "glresource/mi_gl_resource_manager_container.h"
#include "glresource/mi_gl_resource_manager.h"
#include "glresource/mi_gl_texture_1d.h"
#include "glresource/mi_gl_texture_2d.h"
#include "glresource/mi_gl_time_query.h"
#include "glresource/mi_gl_utils.h"

#include "renderalgo/mi_camera_calculator.h"
#include "renderalgo/mi_camera_interactor.h"
#include "renderalgo/mi_color_transfer_function.h"
#include "renderalgo/mi_mpr_entry_exit_points.h"
#include "renderalgo/mi_opacity_transfer_function.h"
#include "renderalgo/mi_ray_caster.h"
#include "renderalgo/mi_ray_caster_canvas.h"
#include "renderalgo/mi_transfer_function_loader.h"
#include "renderalgo/mi_volume_infos.h"
#include "renderalgo/mi_vr_scene.h"
#include "renderalgo/mi_brick_pool.h"
#include "renderalgo/mi_render_algo_logger.h"
#include "renderalgo/mi_vr_entry_exit_points.h"

#ifdef WIN32
#include "GL/glut.h"
#else
#include "GL/freeglut.h"
//#include "cuda_runtime.h"
#include <libgen.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#endif

#include <cuda_runtime.h>
#include <cuda.h>  
#include <cuda_gl_interop.h>
#include <cuda_texture_types.h>


#include "libgpujpeg/gpujpeg.h"
#include "libgpujpeg/gpujpeg_common.h"
#include "libgpujpeg/gpujpeg_encoder.h"

#include "mi_cuda_vr_common.h"

extern "C"
int ray_cast(cudaGLTextureReadOnly& entry_tex, cudaGLTextureReadOnly& exit_tex, int width, int height,
    cudaVolumeInfos volume_info, cudaRayCastInfos ray_cast_info, unsigned char* d_result, cudaGLTextureWriteOnly& canvas_tex, bool d_cpy);

extern "C" 
int init_data(cudaVolumeInfos& cuda_volume_infos, unsigned short* data);

extern "C"
int init_mask(cudaVolumeInfos& cuda_volume_infos, unsigned char* data);

extern "C" 
int init_wl_nonmask(cudaRayCastInfos& ray_cast_infos, float* wl_array_norm);

extern "C" 
int init_lut_nonmask(cudaRayCastInfos& ray_cast_infos, unsigned char* lut_array, int lut_length);

extern "C" 
int init_material_nonmask(cudaRayCastInfos& ray_cast_infos, float* material_array);

extern "C"
void ray_tracing_quad_vertex_mapping(Viewport viewport, int width, int height,
    mat4 mat_viewmodel, mat4 mat_projection_inv, mat4 matmvp,
    int vertex_count, float3* d_vertex, float2* d_tex_coordinate, cudaGLTextureReadOnly& mapping_tex,
    unsigned char* d_result, cudaGLTextureWriteOnly& canvas_tex, bool blend);

extern "C"
int rgba8_to_rgb8(int width, int height, unsigned char* d_rgba, unsigned char* d_rgb);

extern "C"
int rgba8_to_rgb8_mirror(int width, int height, unsigned char*  d_rgba, unsigned char*  d_rgb);

using namespace medical_imaging;

class Navigator {
public:
    Navigator() {

    }

    ~Navigator() {

    }

    void set_vr_camera(std::shared_ptr<CameraBase> camera) {
        _camera = camera;
    }

    void init() {
        init_texture();
        init_graphic();
    }

    void init_graphic() {
        const float w = 0.6f;
        const float x_step = 0.33333f;
        const float y_step = 0.5f;

        //---------------------------------//
        //triangle
        float vertex[] = {
            -w, -w, w,
            w, -w, w,
            w, w, w,
            -w, w, w,
            -w, -w, -w,
            -w, w, -w,
            w, w, -w,
            w, -w, -w,
            -w, -w, -w,
            -w, -w, w,
            -w, w, w,
            -w, w, -w,
            w, -w, -w,
            w, w, -w,
            w, w, w,
            w, -w, w,
            -w, w, -w,
            -w, w, w,
            w, w, w,
            w, w, -w,
            -w, -w, -w,
            w, -w, -w,
            w, -w, w,
            -w, -w, w
        };

        float tex_coordinate[] = {
            x_step*2.0f, y_step,
            x_step*3.0f, y_step,
            x_step*3.0f, 0,
            x_step*2.0f, 0,
            x_step*2.0f, y_step*2.0f,
            x_step*2.0f, y_step,
            x_step*3.0f, y_step,
            x_step*3.0f, y_step*2.0f,
            x_step, 0,
            x_step, y_step,
            0, y_step,
            0, 0,
            0, y_step,
            x_step, y_step,
            x_step, y_step * 2,
            0, y_step * 2,
            x_step*2.0f, 0,
            x_step*2.0f, y_step,
            x_step, y_step,
            x_step, 0,
            x_step, y_step,
            x_step*2.0f, y_step,
            x_step*2.0f, y_step*2.0f,
            x_step, y_step*2.0f
        };

        cudaMalloc(&_d_vertex, sizeof(vertex));
        cudaMemcpy(_d_vertex, vertex, sizeof(vertex), cudaMemcpyHostToDevice);

        cudaMalloc(&_d_tex_coordinate, sizeof(tex_coordinate));
        cudaMemcpy(_d_tex_coordinate, tex_coordinate, sizeof(tex_coordinate), cudaMemcpyHostToDevice);
    }

    void init_texture() {
        //texture
#ifdef WIN32
        const std::string navi_img_file("./config/resource/navi_384_256_3.raw");
#else
        const std::string navi_img_file("../config/resource/navi_384_256_3.raw");
#endif
        const unsigned int img_size = 384 * 256 * 3;
        unsigned char* img_buffer = new unsigned char[img_size];
        if (0 != FileUtil::read_raw(navi_img_file, img_buffer, img_size)) {
            MI_LOG(MI_FATAL) << "load navigator image failed.";
        }
        else {
            _navigator_tex = GLResourceManagerContainer::instance()->get_texture_2d_manager()->create_object("navigator");
            _navigator_tex->initialize();
            _navigator_tex->set_description("navigator texture");
            _navigator_tex->bind();
            GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
            GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
            unsigned char* rgba4 = new unsigned char[384 * 256 * 4];
            for (int i = 0; i < 384 * 256; ++i) {
                rgba4[i * 4] = img_buffer[i * 3];
                rgba4[i * 4 + 1] = img_buffer[i * 3 + 1];
                rgba4[i * 4 + 2] = img_buffer[i * 3 + 2];
                rgba4[i * 4 + 3] = 255;
            }
            delete[] img_buffer;
            img_buffer = nullptr;
            _navigator_tex->load(GL_RGBA8, 384, 256, GL_RGBA, GL_UNSIGNED_BYTE, (char*)rgba4);
            delete[] rgba4;
            rgba4 = nullptr;

            _cuda_navagator_tex.target = GL_TEXTURE_2D;
            _cuda_navagator_tex.gl_tex_id = _navigator_tex->get_id();
            register_image(_cuda_navagator_tex);
            map_image(_cuda_navagator_tex);
            bind_texture(_cuda_navagator_tex, cudaReadModeNormalizedFloat, cudaFilterModeLinear, true);
            unmap_image(_cuda_navagator_tex);
            _navigator_tex->unbind();
        }
    }

    void render(Viewport view_port, int canvas_width, int canvas_height, unsigned char* d_canvas, cudaGLTextureWriteOnly cuda_canvas_tex) {

        OrthoCamera camera;
        if (_camera) {
            Vector3 view = _camera->get_view_direction();
            Vector3 up = _camera->get_up_direction();
            camera.set_look_at(Point3::S_ZERO_POINT);
            Point3 eye = Point3::S_ZERO_POINT - view*3;
            camera.set_eye(eye);
            camera.set_up_direction(up);
            camera.set_ortho(-1, 1, -1, 1, 1, 5);
        }

        Matrix4 mat_v = camera.get_view_matrix();
        Matrix4 mat_p = camera.get_projection_matrix(); 
        Matrix4 mat_mvp = mat_p*mat_v;
        Matrix4 mat_pi = mat_p.get_inverse();
        mat4 mat4_v = matrix4_to_mat4(mat_v);
        mat4 mat4_pi = matrix4_to_mat4(mat_pi);
        mat4 mat4_mvp = matrix4_to_mat4(mat_mvp);

        map_image(_cuda_navagator_tex);
        ray_tracing_quad_vertex_mapping(view_port, canvas_width, canvas_height, mat4_v, mat4_pi, mat4_mvp, 24, (float3*)_d_vertex, (float2*)_d_tex_coordinate, _cuda_navagator_tex, d_canvas, cuda_canvas_tex, true);
        unmap_image(_cuda_navagator_tex);

    }

private:
    std::shared_ptr<GLTexture2D> _navigator_tex;
    cudaGLTextureReadOnly _cuda_navagator_tex;
    std::shared_ptr<CameraBase> _camera;

    float3* _d_vertex;
    float2* _d_tex_coordinate;
};


namespace {
    std::shared_ptr<ImageDataHeader> _data_header;
    std::shared_ptr<ImageData> _volume_data;
    std::shared_ptr<VolumeInfos> _volume_infos;
    std::shared_ptr<VREntryExitPoints> _entry_exit_points;
    std::shared_ptr<OrthoCamera> _camera;
    std::shared_ptr<OrthoCameraInteractor> _camera_interactor;
    std::shared_ptr<GLTexture2D> _canvas_tex;

    //CUDA resource
    cudaVolumeInfos  _cuda_volume_infos;
    cudaRayCastInfos _ray_cast_infos;

    cudaGLTextureWriteOnly _cuda_canvas_tex;
    cudaGLTextureReadOnly _cuda_entry_points;
    cudaGLTextureReadOnly _cuda_exit_points;
    
    //Ray tracing navigator
    Navigator _navigator;

    //GPUJPEG for encode result to JPEG
    gpujpeg_encoder* _gpujpeg_encoder_hd = nullptr;             // jpeg encoder HD
    gpujpeg_encoder* _gpujpeg_encoder_ld = nullptr;             // jpeg encoder LD
    gpujpeg_encoder_input _gpujpeg_encoder_input_hd;  // jpeg encoding input HD
    gpujpeg_encoder_input _gpujpeg_encoder_input_ld;  // jpeg encoding input LD


    unsigned char* _cuda_d_canvas = nullptr;

    float _ww = 1500.0f;
    float _wl = -400.0f;

    int _width = 1024;
    int _height = 1024;
    int _button = -1;
    int _button_status = -1;
    Point2 _pre_pos;

    std::shared_ptr<GLTexture2D> _tex_entry_points;

    bool _show_navigator = true;

    unsigned char* _image_buffer_jpeg = new unsigned char[_width*_height * 4];


#ifdef WIN32
    const std::string root = "E:/data/MyData/demo/lung/";
#else
    const std::string root = "/home/wangrui22/data/demo/lung/";
#endif

    std::vector<std::string> GetFiles() {
        std::vector<std::string> files;
        std::set<std::string> dcm_postfix;
        dcm_postfix.insert(".dcm");
        FileUtil::get_all_file_recursion(root + "/LIDC-IDRI-1002", dcm_postfix, files);
        return files;
    }

    void Finalize() {
        _data_header.reset();
        _volume_data.reset();
        _volume_infos.reset();
    }

    template <typename SrcType, typename DstType>
    std::unique_ptr<DstType[]> signed_to_unsigned(unsigned int length,
        double min_gray, void* data_src) {
        std::unique_ptr<DstType[]> data_dst(new DstType[length]);
        SrcType* data_src0 = (SrcType*)(data_src);

        for (unsigned int i = 0; i < length; ++i) {
            data_dst[i] =
                static_cast<DstType>(static_cast<double>(data_src0[i]) - min_gray);
        }

        return std::move(data_dst);
    }
}

void set_wl(cudaRayCastInfos& ray_cast_infos, int mask_level, float* h_wl_array) {
    if (!ray_cast_infos.d_wl_array) {
        cudaMalloc(&ray_cast_infos.d_wl_array, sizeof(float)*mask_level*2);
    }
    cudaMemcpy(ray_cast_infos.d_wl_array, h_wl_array, sizeof(float)*mask_level*2, cudaMemcpyHostToDevice);
}

//cudaTextureObject_t create_lut(std::shared_ptr<ColorTransFunc> color, std::shared_ptr<OpacityTransFunc> opacity) {
//    std::vector<ColorTFPoint> color_pts;
//    color->set_width(S_TRANSFER_FUNC_WIDTH);
//    color->get_point_list(color_pts);
//
//    std::vector<OpacityTFPoint> opacity_pts;
//    opacity->set_width(S_TRANSFER_FUNC_WIDTH);
//    opacity->get_point_list(opacity_pts);
//
//    unsigned char* rgba = new unsigned char[S_TRANSFER_FUNC_WIDTH * 4];
//
//    for (int i = 0; i < S_TRANSFER_FUNC_WIDTH; ++i) {
//        rgba[i * 4] = static_cast<unsigned char>(color_pts[i].x);
//        rgba[i * 4 + 1] = static_cast<unsigned char>(color_pts[i].y);
//        rgba[i * 4 + 2] = static_cast<unsigned char>(color_pts[i].z);
//        rgba[i * 4 + 3] = static_cast<unsigned char>(opacity_pts[i].a);
//    }
//    
//}

void init_data() {
    Configure::instance()->set_processing_unit_type(GPU);
    GLUtils::set_check_gl_flag(true);
#ifdef WIN32
    Logger::instance()->bind_config_file("./config/log_config");
#else
    Logger::instance()->bind_config_file("../config/log_config");
#endif

    Logger::instance()->initialize();

    //Volume Data
    std::vector<std::string> files = GetFiles();
    DICOMLoader loader;
    loader.load_series(files, _volume_data, _data_header);
    const unsigned int data_len = _volume_data->_dim[0] * _volume_data->_dim[1] * _volume_data->_dim[2];

    //Mask Data
    // Create empty mask
    std::shared_ptr<ImageData> mask_data(new ImageData());
    _volume_data->shallow_copy(mask_data.get());
    mask_data->_channel_num = 1;
    mask_data->_data_type = medical_imaging::UCHAR;
    mask_data->mem_allocate();
    char* mask_raw = (char*)mask_data->get_pixel_pointer();
    std::ifstream in(root + "/mask.raw", std::ios::in);
    if (in.is_open()) {
        in.read(mask_raw, data_len);
        in.close();
    }
    else {
        memset(mask_raw, 1, data_len);
    }
    
    _volume_infos.reset(new VolumeInfos(GPU_BASE, GL_BASE));
    _volume_infos->set_data_header(_data_header);
    _volume_infos->set_volume(_volume_data);
    _volume_infos->set_mask(mask_data);
    std::vector<unsigned char> visible_labels;
    visible_labels.push_back(1);
    _volume_infos->get_brick_pool()->add_visible_labels_cache(visible_labels);
    _volume_infos->refresh();

    _cuda_volume_infos.dim.x = _volume_data->_dim[0];
    _cuda_volume_infos.dim.y = _volume_data->_dim[1];
    _cuda_volume_infos.dim.z = _volume_data->_dim[2];
    _cuda_volume_infos.dim_r = make_float3(1.0f / (float)_volume_data->_dim[0], 1.0f / (float)_volume_data->_dim[1], 1.0f / (float)_volume_data->_dim[2]);
    _cuda_volume_infos.sample_shift = 0.5f *  _cuda_volume_infos.dim_r *
        make_float3(_volume_data->_spacing[0], _volume_data->_spacing[1], _volume_data->_spacing[2]);
    

    //Create CUDA Texture
    if (_volume_data->_data_type == DataType::SHORT) {
        std::unique_ptr<unsigned short[]> dst_data = signed_to_unsigned<short, unsigned short>(
            data_len, _volume_data->get_min_scalar(), _volume_data->get_pixel_pointer());
        init_data(_cuda_volume_infos, dst_data.get());
    }
    else {
        init_data(_cuda_volume_infos, (unsigned short*)_volume_data->get_pixel_pointer());
    }
    init_mask(_cuda_volume_infos, (unsigned char*)mask_data->get_pixel_pointer());

    

    //LUT
    std::shared_ptr<ColorTransFunc> color;
    std::shared_ptr<OpacityTransFunc> opacity;
    float ww, wl;
    RGBAUnit background;
    Material material;
#ifdef WIN32
    std::string color_opacity_xml = "../../../config/lut/3d/ct_cta.xml";
#else
    std::string color_opacity_xml = "../config/lut/3d/ct_cta.xml";
#endif
    if (IO_SUCCESS !=
        TransferFuncLoader::load_color_opacity(color_opacity_xml, color, opacity,
            ww, wl, background, material)) {
        MI_LOG(MI_ERROR) << "load LUT : " << color_opacity_xml << " failed.";
    }
    _ww = ww;
    _wl = wl;

    std::vector<ColorTFPoint> color_pts;
    color->set_width(S_TRANSFER_FUNC_WIDTH);
    color->get_point_list(color_pts);

    std::vector<OpacityTFPoint> opacity_pts;
    opacity->set_width(S_TRANSFER_FUNC_WIDTH);
    opacity->get_point_list(opacity_pts);

    unsigned char* rgba = new unsigned char[S_TRANSFER_FUNC_WIDTH * 4];

    for (int i = 0; i < S_TRANSFER_FUNC_WIDTH; ++i) {
        rgba[i * 4] = static_cast<unsigned char>(color_pts[i].x);
        rgba[i * 4 + 1] = static_cast<unsigned char>(color_pts[i].y);
        rgba[i * 4 + 2] = static_cast<unsigned char>(color_pts[i].z);
        rgba[i * 4 + 3] = static_cast<unsigned char>(opacity_pts[i].a);
    }
    init_lut_nonmask(_ray_cast_infos, rgba, S_TRANSFER_FUNC_WIDTH);

    //Materials
    float material_array[9] = {
        material.diffuse[0],material.diffuse[1],material.diffuse[2],material.diffuse[3],
        material.specular[0],material.specular[1],material.specular[2],material.specular[3],
        material.specular_shiness};
    init_material_nonmask(_ray_cast_infos, material_array);

    //WL
    _volume_data->regulate_wl(ww, wl);
    _volume_data->normalize_wl(ww, wl);
    float wl_array[2] = { ww,wl };
    init_wl_nonmask(_ray_cast_infos, wl_array);

    //Sample Step
    _ray_cast_infos.sample_step = 0.5f;

    //navigator
    _navigator.init();

    MI_LOG(MI_INFO) << "init data success.";
}

void init_gl() {
    //Global GL state
    GLUtils::set_check_gl_flag(true);
    GLUtils::set_pixel_pack_alignment(1);

    //Entry exit points
    _camera.reset(new OrthoCamera());
    std::shared_ptr<CameraCalculator> camera_cal = _volume_infos->get_camera_calculator();
    camera_cal->init_vr_placement(_camera);
    _camera_interactor.reset(new OrthoCameraInteractor(_camera));
    _camera_interactor->reset_camera();
    _camera_interactor->resize(_width, _height);

    _entry_exit_points.reset(new VREntryExitPoints(GPU_BASE, GL_BASE));
    _entry_exit_points->set_display_size(_width, _height);
    _entry_exit_points->initialize();
    _entry_exit_points->set_proxy_geometry(PG_BRICKS);
    _entry_exit_points->set_camera(_camera);
    _entry_exit_points->set_camera_calculator(camera_cal);
    std::vector<unsigned char> visible_labels;
    visible_labels.push_back(1);
    _entry_exit_points->set_visible_labels(visible_labels);

    std::shared_ptr<ImageData> volume = _volume_infos->get_volume();
    _entry_exit_points->set_volume_data(volume);
    _entry_exit_points->set_brick_pool(_volume_infos->get_brick_pool());
    AABB default_aabb;
    default_aabb._min = Point3::S_ZERO_POINT;
    default_aabb._max.x = static_cast<double>(volume->_dim[0]);
    default_aabb._max.y = static_cast<double>(volume->_dim[1]);
    default_aabb._max.z = static_cast<double>(volume->_dim[2]);
    _entry_exit_points->set_bounding_box(default_aabb);
    _entry_exit_points->set_brick_pool(_volume_infos->get_brick_pool());

    _entry_exit_points->set_brick_filter_item(BF_MASK | BF_WL);
    std::map<unsigned char, Vector2f> wls;
    wls.insert(std::make_pair(1, Vector2f(_ww,_wl)));
    _entry_exit_points->set_window_levels(wls);

    _tex_entry_points = _entry_exit_points->get_entry_points_texture()->get_gl_resource();


    //Canvas texture
    glEnable(GL_TEXTURE_2D);
    _canvas_tex = GLResourceManagerContainer::instance()->get_texture_2d_manager()->create_object("canvas");
    _canvas_tex->set_description("CUDA GL UT canvas texture.");
    _canvas_tex->initialize();
    _canvas_tex->bind();
    GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_EDGE);
    GLTextureUtils::set_filter(GL_TEXTURE_2D, GL_LINEAR);
    _canvas_tex->load(GL_RGBA8, _width, _height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    //CUDA canvas
    _cuda_canvas_tex.gl_tex_id = _canvas_tex->get_id();
    _cuda_canvas_tex.target = GL_TEXTURE_2D;
    register_image(_cuda_canvas_tex);
    cudaMalloc(&_cuda_d_canvas, _width*_height * 4);
    _canvas_tex->unbind();

    //generate CUDA texture
    _entry_exit_points->get_entry_points_texture()->get_gl_resource()->bind();
    _cuda_entry_points.gl_tex_id = _entry_exit_points->get_entry_points_texture()->get_gl_resource()->get_id();
    _cuda_entry_points.target = GL_TEXTURE_2D;
    register_image(_cuda_entry_points);
    map_image(_cuda_entry_points);
    bind_texture(_cuda_entry_points, cudaReadModeElementType, cudaFilterModePoint, false);
    unmap_image(_cuda_entry_points);

    _entry_exit_points->get_exit_points_texture()->get_gl_resource()->bind();
    _cuda_exit_points.gl_tex_id = _entry_exit_points->get_exit_points_texture()->get_gl_resource()->get_id();
    _cuda_exit_points.target = GL_TEXTURE_2D;
    register_image(_cuda_exit_points);
    map_image(_cuda_exit_points);
    bind_texture(_cuda_exit_points, cudaReadModeElementType, cudaFilterModePoint, false);
    unmap_image(_cuda_exit_points);

    if (cudaSuccess != cudaMalloc(&_cuda_d_canvas, _width*_height * 4)) {
        MI_LOG(MI_ERROR) << "[CUDA] " << "malloc canvas device memory failed.";
    }
}

void init_gpujpeg() {
    //gpujpeg_init_device(0,0);

    const int _compress_hd_quality = 80;
    const int _compress_ld_quality = 15;

    gpujpeg_parameters gpujpeg_param_hd;
    gpujpeg_set_default_parameters(&gpujpeg_param_hd);        //默认参数
    gpujpeg_parameters_chroma_subsampling(&gpujpeg_param_hd); //默认采样参数;
    gpujpeg_param_hd.quality = _compress_hd_quality;

    gpujpeg_parameters gpujpeg_param_ld;
    gpujpeg_set_default_parameters(&gpujpeg_param_ld);        //默认参数
    gpujpeg_parameters_chroma_subsampling(&gpujpeg_param_ld); //默认采样参数;
    gpujpeg_param_ld.quality = _compress_ld_quality;

    gpujpeg_image_parameters gpujpeg_image_param;
    gpujpeg_image_set_default_parameters(&gpujpeg_image_param);
    gpujpeg_image_param.width = _width;
    gpujpeg_image_param.height = _height;
    gpujpeg_image_param.comp_count = 3;
    gpujpeg_image_param.color_space = GPUJPEG_RGB;
    gpujpeg_image_param.sampling_factor = GPUJPEG_4_4_4;

    _gpujpeg_encoder_hd = gpujpeg_encoder_create(&gpujpeg_param_hd, &gpujpeg_image_param);
    _gpujpeg_encoder_ld = gpujpeg_encoder_create(&gpujpeg_param_ld, &gpujpeg_image_param);
    gpujpeg_image_destroy(_gpujpeg_encoder_input_hd.image);
    gpujpeg_image_destroy(_gpujpeg_encoder_input_ld.image);

    _gpujpeg_encoder_input_ld.type = GPUJPEG_ENCODER_INPUT_INTERNAL_BUFFER;
    _gpujpeg_encoder_input_hd.type = GPUJPEG_ENCODER_INPUT_INTERNAL_BUFFER;
}

inline void jpeg_encode(unsigned char* d_array_rgba8, bool downsample) {
    if (downsample) {
        unsigned char* rgb = (unsigned char*)gpujpeg_encoder_get_inner_device_image_data(_gpujpeg_encoder_ld);
        rgba8_to_rgb8_mirror(_width, _height, d_array_rgba8, rgb);

        uint8_t* image_compressed = nullptr;
        int image_compressed_size = 0;
        if(0 !=  gpujpeg_encoder_encode(_gpujpeg_encoder_ld, &_gpujpeg_encoder_input_ld,
            &image_compressed, &image_compressed_size) ) {
            MI_LOG(MI_ERROR) << "GPUJPEG encode failed.";
            return;
        }

        memcpy((char*)_image_buffer_jpeg, image_compressed, image_compressed_size);
//#ifdef WIN32
//        FileUtil::write_raw("D:/temp/cuda_output_ut_ld.jpeg", image_compressed, image_compressed_size);
//#else
//        FileUtil::write_raw("/home/wangrui22/data/cuda_output_ut_ld.jpeg", image_compressed, image_compressed_size);
//#endif
    }
    else {
        unsigned char* rgb = (unsigned char*)gpujpeg_encoder_get_inner_device_image_data(_gpujpeg_encoder_hd);
        rgba8_to_rgb8_mirror(_width, _height, d_array_rgba8, rgb);

        uint8_t* image_compressed = nullptr;
        int image_compressed_size = 0;
        if (0 != gpujpeg_encoder_encode(_gpujpeg_encoder_hd, &_gpujpeg_encoder_input_hd,
            &image_compressed, &image_compressed_size)) {
            MI_LOG(MI_ERROR) << "GPUJPEG encode failed.";
            return;
        }

        memcpy((char*)_image_buffer_jpeg, image_compressed, image_compressed_size);
//#ifdef WIN32
//        FileUtil::write_raw("D:/temp/cuda_output_ut_hd.jpeg", image_compressed, image_compressed_size);
//#else
//        FileUtil::write_raw("/home/wangrui22/data/cuda_output_ut_hd.jpeg", image_compressed, image_compressed_size);
//#endif
    }

    
}

void Display() {
    try {
        CHECK_GL_ERROR;

        //Quat4 q(5.0 / 360.0 * 2.0 * 3.1415926, Vector3(0, 1, 0));
        //_camera->rotate(q);

        //calculate entry exit points
        _entry_exit_points->calculate_entry_exit_points();

        //CUDA process
        //calculate normal matrix
        std::shared_ptr<CameraCalculator> camera_cal = _volume_infos->get_camera_calculator();
        const Matrix4 mat_v2w = camera_cal->get_volume_to_world_matrix();
        const Matrix4 mat_view = _camera->get_view_matrix();
        const Matrix4 mat_normal = (mat_view * mat_v2w).get_inverse().get_transpose();
        _ray_cast_infos.mat_normal = matrix4_to_mat4(mat_normal);
        //calculate light position
        Point3 eye = _camera->get_eye();
        Point3 lookat = _camera->get_look_at();
        Vector3 view = _camera->get_view_direction();
        const static double max_dim =
            (std::max)((std::max)(_volume_infos->get_volume()->_dim[0] * _volume_infos->get_volume()->_spacing[0],
                _volume_infos->get_volume()->_dim[1] * _volume_infos->get_volume()->_spacing[1]),
                _volume_infos->get_volume()->_dim[2] * _volume_infos->get_volume()->_spacing[2]);
        const static float magic_num = 1.5f;
        Point3 light_pos = lookat - view * max_dim * magic_num;
        light_pos = camera_cal->get_world_to_volume_matrix().transform(light_pos);
        _ray_cast_infos.light_position = make_float3(light_pos.x, light_pos.y, light_pos.z);

        //ray cast
        ray_cast(_cuda_entry_points, _cuda_exit_points, _width, _height, _cuda_volume_infos, _ray_cast_infos, _cuda_d_canvas, _cuda_canvas_tex, !_show_navigator);
        
        CHECK_GL_ERROR;

        if(_show_navigator) {
            const int navigator_margin = 20; 
            const float navigator_window_ratio = 4.5f;
            const int min_size = int((std::min)(_width, _height) / navigator_window_ratio);
            Viewport view_port(_width - min_size - navigator_margin, navigator_margin, min_size, min_size);
            _navigator.set_vr_camera(_camera);
            _navigator.render(view_port, _width, _height, _cuda_d_canvas, _cuda_canvas_tex);
        }

        jpeg_encode(_cuda_d_canvas, true);

        glViewport(0, 0, _width, _height);
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindTexture(GL_TEXTURE_2D, _canvas_tex->get_id());
        //glBindTexture(GL_TEXTURE_2D, _entry_exit_points->get_entry_points_texture()->get_id());
        glBegin(GL_QUADS);
        glTexCoord2d(0, 0);
        glVertex2d(-1.0, -1.0);
        glTexCoord2d(1, 0);
        glVertex2d(1.0, -1.0);
        glTexCoord2d(1, 1);
        glVertex2d(1.0, 1.0);
        glTexCoord2d(0, 1);
        glVertex2d(-1.0, 1.0);
        glEnd();

        glutSwapBuffers();
    }
    catch (Exception& e) {
        MI_RENDERALGO_LOG(MI_ERROR) << e.what();
        abort();
    }
}

void Keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 'n' :
        {
            _show_navigator = !_show_navigator;
           break;
        }
    default:
        break;
    }

    glutPostRedisplay();
}

void Resize(int x, int y) {
    if (x == 0 || y == 0) {
        return;
    }

    _width = x;
    _height = y;
    glutPostRedisplay();
}

void Idle() {
    glutPostRedisplay();
}

void MouseClick(int button, int status, int x, int y) {
    _button = button;
    _button_status = status;
    _pre_pos = Point2(x, y);
    glutPostRedisplay();
}

void MouseMotion(int x, int y) {
    Point2 pt(x, y);
    if (_button_status == GLUT_DOWN) {
        if (_button == GLUT_LEFT_BUTTON) {
            _camera_interactor->rotate(_pre_pos, pt, _width, _height);
        } else if (_button == GLUT_RIGHT_BUTTON) {
            _camera_interactor->zoom(_pre_pos, pt, _width, _height);
        } else if (_button == GLUT_MIDDLE_BUTTON) {
            _ww += (float)(x - _pre_pos.x);
            _wl += (float)(_pre_pos.y - y);
            _ww = _ww < 1.0f ? 1.0f : _ww;
            std::map<unsigned char, Vector2f> wls;
            wls.insert(std::make_pair(1, Vector2f(_ww,_wl)));
            _entry_exit_points->set_window_levels(wls);
        }
    }

    _pre_pos = pt;
    glutPostRedisplay();
}

int mi_cuda_vr(int argc, char* argv[]) {
#ifndef WIN32
    chdir(dirname(argv[0]));
#endif

    try {
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
        glutInitWindowPosition(0, 0);
        glutInitWindowSize(_width, _height);

        glutCreateWindow("CUDA VR");

        if (GLEW_OK != glewInit()) {
            MI_RENDERALGO_LOG(MI_FATAL) << "Init GLEW failed!\n";
            return -1;
        }

        GLEnvironment env;
        int major, minor;
        env.get_gl_version(major, minor);

        init_data();
        init_gl();
        init_gpujpeg();

        glutDisplayFunc(Display);
        glutReshapeFunc(Resize);
        glutIdleFunc(Idle);
        glutKeyboardFunc(Keyboard);
        glutMouseFunc(MouseClick);
        glutMotionFunc(MouseMotion);

        glutMainLoop();

        Finalize();

        return 0;
    }
    catch (const Exception& e) {
        MI_RENDERALGO_LOG(MI_FATAL) << e.what();
        return -1;
    }
}