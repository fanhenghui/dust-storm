ver = release
CC  = g++ -std=c++11 -Wall -Wno-unused-function
NVCC = nvcc -ccbin g++ -std c++11 \
	-gencode=arch=compute_30,code=sm_30 \
 	-gencode=arch=compute_50,code=sm_50 \
 	-gencode=arch=compute_52,code=sm_52 \
 	-gencode=arch=compute_60,code=sm_60 \
 	-gencode=arch=compute_61,code=sm_61 \
 	-gencode=arch=compute_61,code=compute_61 \
	-O3

ifeq ($(ver) , release)
CFLAGS += -O3 -fopenmp
else
CFLAGS += -g -fopenmp
endif

DEFS = -DBOOST_LOG_DYN_LINK
CFLAGS += $(DEFS)

ROOT = ..
INCLUDE_ROOT_DIR = $(ROOT)/include
INCLUDE_DIR = $(INCLUDE_ROOT_DIR)/renderalgo
CUDA_INCLUDE_DIR = $(CUDA_HOME)/include
BIN_DIR = $(ROOT)/bin

LOCAL_LIB_DIR = /usr/local/lib
CUDA_LIB_DIR = $(CUDA_HOME)/lib64
VPATH = $(INCLUDE_ROOT_DIR) : $(INCLUDE_DIR)

#GPU render step
GPU_REBDER_STEP = mi_rc_step_color_inverse.o mi_rc_step_composite.o mi_rc_step_main.o mi_rc_step_mask_overlay.o \
mi_rc_step_mask_sampler.o mi_rc_step_ray_casting.o mi_rc_step_shading.o mi_rc_step_utils.o mi_rc_step_volume_sampler.o

#CUDA kernel
KERNEL_DIR = ./kernel/
CUDA_KERNEL = kernel_ray_tracing.o kernel_canvas_transformer.o kernel_mpr_entry_exit_points.o kernel_brick_info_calculator.o kernel_ray_casting.o 
CUDA_KERNEL_LINK =  kernel_link.o 

OBJS = $(CUDA_KERNEL) $(CUDA_KERNEL_LINK) mi_brick_info_calculator.o mi_brick_pool.o mi_mask_label_store.o \
mi_camera_calculator.o  mi_camera_interactor.o \
mi_color_transfer_function.o mi_opacity_transfer_function.o mi_transfer_function_loader.o \
mi_entry_exit_points.o mi_mpr_entry_exit_points.o mi_vr_entry_exit_points.o mi_vr_proxy_geometry_cube.o mi_vr_proxy_geometry_brick.o\
mi_ray_caster_canvas.o mi_ray_caster_inner_resource.o mi_ray_caster.o mi_ray_casting_cpu.o mi_ray_casting_gpu_gl.o mi_ray_casting_gpu_cuda.o \
$(GPU_REBDER_STEP) \
mi_volume_infos.o mi_scene_base.o mi_mpr_scene.o mi_ray_cast_scene.o mi_vr_scene.o \
mi_graphic_object_navigator.o mi_gpu_image_compressor.o mi_transfer_function_texture.o \
mi_render_global_resource.o

TARGET = $(BIN_DIR)/libmedimg_renderalgo.so

ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) -I$(INCLUDE_DIR) -I$(CUDA_INCLUDE_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -L$(LOCAL_LIB_DIR) -L$(CUDA_LIB_DIR) -lpthread -lboost_system -lboost_filesystem -lboost_thread -lboost_log -lGL -lGLEW -lcudart \
-lmedimg_log -lmedimg_util -lmedimg_arithmetic -lmedimg_glresource -lmedimg_cudaresource -lmedimg_io \
-lgpujpeg -lpugixml 

all: pre-build main-build

pre-build:
	../bin/shaderencoder ./shader/shader_config

main-build: $(TARGET)

$(TARGET): $(OBJS) $(BIN_DIR)/libmedimg_util.so $(BIN_DIR)/libmedimg_glresource.so $(BIN_DIR)/libmedimg_io.so
	$(CC) -fPIC -DPIC -shared $(CFLAGS) -o $(TARGET) $(OBJS) $(ADDITIANL_DEPENDENCIES)

#Kernel
$(CUDA_KERNEL) :%.o: $(KERNEL_DIR)%.cu 
	$(NVCC) -Xcompiler -fPIC  -dc $< -I$(INCLUDE_ROOT_DIR)

$(CUDA_KERNEL_LINK) : $(CUDA_KERNEL)
	$(NVCC) -shared -dlink $(CUDA_KERNEL) -o $(CUDA_KERNEL_LINK)

#scene related
mi_gpu_image_compressor.o: mi_gpu_image_compressor.h mi_gpu_image_compressor.cpp $(CUDA_KERNEL_LINK)
	$(CC) -c -fPIC $(CFLAGS) mi_gpu_image_compressor.cpp $(ADDITINAL_INCLUDE_DIR)

mi_vr_scene.o: mi_vr_scene.h mi_vr_scene.cpp mi_ray_cast_scene.o \
mi_color_transfer_function.o mi_opacity_transfer_function.o mi_camera_calculator.o mi_camera_interactor.o \
mi_volume_infos.o mi_vr_entry_exit_points.o mi_render_global_resource.o
	$(CC) -c -fPIC $(CFLAGS) mi_vr_scene.cpp $(ADDITINAL_INCLUDE_DIR)

mi_mpr_scene.o: mi_mpr_scene.h mi_mpr_scene.cpp mi_ray_cast_scene.o \
mi_camera_calculator.o mi_camera_interactor.o mi_volume_infos.o mi_mpr_entry_exit_points.o mi_ray_caster.o
	$(CC) -c -fPIC $(CFLAGS) mi_mpr_scene.cpp $(ADDITINAL_INCLUDE_DIR)

mi_ray_cast_scene.o: mi_ray_cast_scene.h mi_ray_cast_scene.cpp mi_scene_base.o \
mi_camera_calculator.o mi_camera_interactor.o mi_volume_infos.o mi_entry_exit_points.o mi_ray_caster.o \
mi_ray_caster_canvas.o mi_graphic_object_navigator.o
	$(CC) -c -fPIC $(CFLAGS) mi_ray_cast_scene.cpp $(ADDITINAL_INCLUDE_DIR)

mi_scene_base.o: mi_scene_base.h mi_scene_base.cpp mi_gpu_image_compressor.o
	$(CC) -c -fPIC $(CFLAGS) mi_scene_base.cpp $(ADDITINAL_INCLUDE_DIR)

mi_volume_infos.o: mi_volume_infos.h mi_volume_infos.cpp \
mi_camera_calculator.o mi_brick_info_calculator.o mi_brick_pool.o 
	$(CC) -c -fPIC $(CFLAGS) mi_volume_infos.cpp $(ADDITINAL_INCLUDE_DIR)

#Ray casting
mi_ray_caster.o: mi_ray_caster.h mi_ray_caster.cpp \
mi_ray_caster_canvas.o mi_ray_casting_cpu.o mi_ray_caster_inner_resource.o mi_ray_casting_gpu_gl.o mi_ray_casting_gpu_cuda.o
	$(CC) -c -fPIC $(CFLAGS) mi_ray_caster.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_ray_casting_gpu_gl.o: mi_ray_casting_gpu_gl.h mi_ray_casting_gpu_gl.cpp \
mi_entry_exit_points.o mi_ray_caster_canvas.o mi_ray_caster_inner_resource.o $(GPU_REBDER_STEP)
	$(CC) -c -fPIC $(CFLAGS) mi_ray_casting_gpu_gl.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_ray_casting_gpu_cuda.o: mi_ray_casting_gpu_cuda.h mi_ray_casting_gpu_cuda.cpp \
mi_entry_exit_points.o mi_ray_caster_canvas.o mi_ray_caster_inner_resource.o $(GPU_REBDER_STEP) $(CUDA_KERNEL_LINK)
	$(CC) -c -fPIC $(CFLAGS) mi_ray_casting_gpu_cuda.cpp $(ADDITINAL_INCLUDE_DIR)	

$(GPU_REBDER_STEP): %.o:%.cpp %.h 
	$(CC) -c -fPIC $(CFLAGS) -Wno-unused-variable -o $@ $< $(ADDITINAL_INCLUDE_DIR)

mi_ray_casting_cpu.o: mi_ray_casting_cpu.h mi_ray_casting_cpu.cpp \
mi_entry_exit_points.o mi_ray_caster_canvas.o
	$(CC) -c -fPIC $(CFLAGS) mi_ray_casting_cpu.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_ray_caster_inner_resource.o: mi_ray_caster_inner_resource.h mi_ray_caster_inner_resource.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_ray_caster_inner_resource.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_ray_caster_canvas.o: mi_ray_caster_canvas.h mi_ray_caster_canvas.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_ray_caster_canvas.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_render_global_resource.o: mi_render_global_resource.h mi_render_global_resource.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_render_global_resource.cpp $(ADDITINAL_INCLUDE_DIR)

#Entry exit points(VR)
mi_vr_entry_exit_points.o: mi_vr_entry_exit_points.h mi_vr_entry_exit_points.cpp mi_ray_caster_define.h \
mi_camera_calculator.o mi_entry_exit_points.o mi_vr_proxy_geometry_cube.o mi_vr_proxy_geometry_brick.o
	$(CC) -c -fPIC $(CFLAGS) mi_vr_entry_exit_points.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_vr_proxy_geometry_cube.o: mi_vr_proxy_geometry_cube.h mi_vr_proxy_geometry_cube.cpp mi_ray_caster_define.h \
mi_camera_calculator.o
	$(CC) -c -fPIC $(CFLAGS) -Wno-unused-variable mi_vr_proxy_geometry_cube.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_vr_proxy_geometry_brick.o: mi_vr_proxy_geometry_brick.h mi_vr_proxy_geometry_brick.cpp mi_ray_caster_define.h \
mi_camera_calculator.o
	$(CC) -c -fPIC $(CFLAGS) -Wno-unused-variable mi_vr_proxy_geometry_brick.cpp $(ADDITINAL_INCLUDE_DIR)		

#Entry exit points(MPR)
mi_mpr_entry_exit_points.o: mi_mpr_entry_exit_points.h mi_mpr_entry_exit_points.cpp mi_ray_caster_define.h \
mi_camera_calculator.o mi_entry_exit_points.o $(CUDA_KERNEL_LINK)
	$(CC) -c -fPIC $(CFLAGS) -Wno-unused-variable mi_mpr_entry_exit_points.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_entry_exit_points.o: mi_entry_exit_points.h mi_entry_exit_points.cpp mi_camera_calculator.o
	$(CC) -c -fPIC $(CFLAGS) mi_entry_exit_points.cpp $(ADDITINAL_INCLUDE_DIR)	

#Transfer function
mi_transfer_function_loader.o: mi_transfer_function_loader.h mi_transfer_function_loader.cpp mi_ray_caster_define.h mi_color_transfer_function.o mi_opacity_transfer_function.o
	$(CC) -c -fPIC $(CFLAGS) mi_transfer_function_loader.cpp $(ADDITINAL_INCLUDE_DIR)

mi_color_transfer_function.o: mi_color_transfer_function.h mi_color_transfer_function.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_color_transfer_function.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_opacity_transfer_function.o: mi_opacity_transfer_function.h mi_opacity_transfer_function.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_opacity_transfer_function.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_transfer_function_texture.o: mi_transfer_function_texture.h mi_transfer_function_texture.cpp mi_opacity_transfer_function.o mi_color_transfer_function.o
	$(CC) -c -fPIC $(CFLAGS) mi_transfer_function_texture.cpp $(ADDITINAL_INCLUDE_DIR)

#Camera related
mi_camera_interactor.o: mi_camera_interactor.h mi_camera_interactor.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_camera_interactor.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_camera_calculator.o: mi_camera_calculator.h mi_camera_calculator.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_camera_calculator.cpp $(ADDITINAL_INCLUDE_DIR)	

#Brick related
mi_brick_pool.o: mi_brick_pool.h mi_brick_pool.cpp mi_brick_define.h mi_brick_info_calculator.o	
	$(CC) -c -fPIC $(CFLAGS) mi_brick_pool.cpp $(ADDITINAL_INCLUDE_DIR)

mi_brick_info_calculator.o: mi_brick_info_calculator.h mi_brick_info_calculator.cpp mi_brick_define.h $(CUDA_KERNEL_LINK)
	$(CC) -c -fPIC $(CFLAGS) -Wno-unused-variable mi_brick_info_calculator.cpp $(ADDITINAL_INCLUDE_DIR)

mi_mask_label_store.o: mi_mask_label_store.h mi_mask_label_store.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_mask_label_store.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_graphic_object_navigator.o: mi_graphic_object_navigator.h mi_graphic_object_navigator.cpp $(CUDA_KERNEL_LINK)
	$(CC) -c -fPIC $(CFLAGS) mi_graphic_object_navigator.cpp $(ADDITINAL_INCLUDE_DIR)	


.PHONY : clean
clean:
	-rm *.o $(TARGET)
