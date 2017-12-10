ver = debug
CC  = g++ -std=c++11 -Wall -Wno-unused-function

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
BIN_DIR = $(ROOT)/bin

CUDA_INCLUDE_DIR = /usr/local/cuda-8.0/include

VPATH = $(INCLUDE_ROOT_DIR) : $(INCLUDE_DIR)

#GPU render step
GPU_REBDER_STEP = mi_rc_step_color_inverse.o mi_rc_step_composite.o mi_rc_step_main.o mi_rc_step_mask_overlay.o \
mi_rc_step_mask_sampler.o mi_rc_step_ray_casting.o mi_rc_step_shading.o mi_rc_step_utils.o mi_rc_step_volume_sampler.o

OBJS = mi_brick_info_calculator.o mi_brick_pool.o mi_mask_label_store.o \
mi_camera_calculator.o  mi_camera_interactor.o \
mi_color_transfer_function.o mi_opacity_transfer_function.o mi_transfer_function_loader.o \
mi_entry_exit_points.o mi_mpr_entry_exit_points.o mi_vr_entry_exit_points.o mi_vr_proxy_geometry_cube.o mi_vr_proxy_geometry_brick.o\
mi_ray_caster_canvas.o mi_ray_caster_inner_buffer.o mi_ray_caster.o mi_ray_casting_cpu.o mi_ray_casting_gpu.o \
$(GPU_REBDER_STEP) \
mi_volume_infos.o mi_scene_base.o mi_mpr_scene.o mi_ray_cast_scene.o mi_vr_scene.o \
mi_annotation_calculator.o mi_graphic_object_navigator.o


TARGET = $(BIN_DIR)/libmedimg_renderalgo.so

EXPORT_HEADER = mi_render_algo_export.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) -I$(INCLUDE_DIR) -I$(CUDA_INCLUDE_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -lpthread -lboost_system -lboost_filesystem -lboost_thread -lboost_log -lGL -lGLEW \
-lmedimg_log -lmedimg_util -lmedimg_arithmetic -lmedimg_glresource -lmedimg_io \
-lgpujpeg -lpugixml 


all: pre-build main-build

pre-build:
	../bin/shaderencoder ./shader/shader_config

main-build: $(TARGET)

$(TARGET): $(OBJS) $(BIN_DIR)/libmedimg_util.so $(BIN_DIR)/libmedimg_glresource.so $(BIN_DIR)/libmedimg_io.so
	$(CC) -shared $(CFLAGS) -o $(TARGET) $(OBJS) $(ADDITIANL_DEPENDENCIES)

mi_annotation_calculator.o: $(EXPORT_HEADER) mi_annotation_calculator.h mi_annotation_calculator.cpp mi_mpr_scene.o mi_camera_calculator.o
	$(CC) -c -fPIC $(CFLAGS) mi_annotation_calculator.cpp $(ADDITINAL_INCLUDE_DIR)
	
#scene related
mi_vr_scene.o: $(EXPORT_HEADER) mi_vr_scene.h mi_vr_scene.cpp mi_ray_cast_scene.o \
mi_color_transfer_function.o mi_opacity_transfer_function.o mi_camera_calculator.o mi_camera_interactor.o mi_volume_infos.o mi_vr_entry_exit_points.o
	$(CC) -c -fPIC $(CFLAGS) mi_vr_scene.cpp $(ADDITINAL_INCLUDE_DIR)

mi_mpr_scene.o: $(EXPORT_HEADER) mi_mpr_scene.h mi_mpr_scene.cpp mi_ray_cast_scene.o \
mi_camera_calculator.o mi_camera_interactor.o mi_volume_infos.o mi_mpr_entry_exit_points.o mi_ray_caster.o
	$(CC) -c -fPIC $(CFLAGS) mi_mpr_scene.cpp $(ADDITINAL_INCLUDE_DIR)

mi_ray_cast_scene.o: $(EXPORT_HEADER) mi_ray_cast_scene.h mi_ray_cast_scene.cpp mi_scene_base.o \
mi_camera_calculator.o mi_camera_interactor.o mi_volume_infos.o mi_entry_exit_points.o mi_ray_caster.o \
mi_ray_caster_canvas.o mi_graphic_object_navigator.o
	$(CC) -c -fPIC $(CFLAGS) mi_ray_cast_scene.cpp $(ADDITINAL_INCLUDE_DIR)

mi_scene_base.o: $(EXPORT_HEADER) mi_scene_base.h mi_scene_base.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_scene_base.cpp $(ADDITINAL_INCLUDE_DIR)

mi_volume_infos.o: $(EXPORT_HEADER) mi_volume_infos.h mi_volume_infos.cpp \
mi_camera_calculator.o mi_brick_info_calculator.o mi_brick_pool.o 
	$(CC) -c -fPIC $(CFLAGS) mi_volume_infos.cpp $(ADDITINAL_INCLUDE_DIR)

#Ray casting
mi_ray_caster.o: $(EXPORT_HEADER) mi_ray_caster.h mi_ray_caster.cpp \
mi_ray_caster_canvas.o mi_ray_casting_cpu.o mi_ray_caster_inner_buffer.o mi_ray_casting_gpu.o
	$(CC) -c -fPIC $(CFLAGS) mi_ray_caster.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_ray_casting_gpu.o: $(EXPORT_HEADER) mi_ray_casting_gpu.h mi_ray_casting_gpu.cpp \
mi_entry_exit_points.o mi_ray_caster_canvas.o $(GPU_REBDER_STEP)
	$(CC) -c -fPIC $(CFLAGS) mi_ray_casting_gpu.cpp $(ADDITINAL_INCLUDE_DIR)	

$(GPU_REBDER_STEP): %.o:%.cpp %.h 
	$(CC) -c -fPIC $(CFLAGS) -Wno-unused-variable -o $@ $< $(ADDITINAL_INCLUDE_DIR)

mi_ray_casting_cpu.o: $(EXPORT_HEADER) mi_ray_casting_cpu.h mi_ray_casting_cpu.cpp \
mi_entry_exit_points.o mi_ray_caster_canvas.o
	$(CC) -c -fPIC $(CFLAGS) mi_ray_casting_cpu.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_ray_caster_inner_buffer.o: $(EXPORT_HEADER) mi_ray_caster_inner_buffer.h mi_ray_caster_inner_buffer.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_ray_caster_inner_buffer.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_ray_caster_canvas.o: $(EXPORT_HEADER) mi_ray_caster_canvas.h mi_ray_caster_canvas.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_ray_caster_canvas.cpp $(ADDITINAL_INCLUDE_DIR)	

#Entry exit points(VR)
mi_vr_entry_exit_points.o: $(EXPORT_HEADER) mi_vr_entry_exit_points.h mi_vr_entry_exit_points.cpp mi_ray_caster_define.h \
mi_camera_calculator.o mi_entry_exit_points.o mi_vr_proxy_geometry_cube.o mi_vr_proxy_geometry_brick.o
	$(CC) -c -fPIC $(CFLAGS) mi_vr_entry_exit_points.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_vr_proxy_geometry_cube.o: $(EXPORT_HEADER) mi_vr_proxy_geometry_cube.h mi_vr_proxy_geometry_cube.cpp mi_ray_caster_define.h \
mi_camera_calculator.o
	$(CC) -c -fPIC $(CFLAGS) -Wno-unused-variable mi_vr_proxy_geometry_cube.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_vr_proxy_geometry_brick.o: $(EXPORT_HEADER) mi_vr_proxy_geometry_brick.h mi_vr_proxy_geometry_brick.cpp mi_ray_caster_define.h \
mi_camera_calculator.o
	$(CC) -c -fPIC $(CFLAGS) -Wno-unused-variable mi_vr_proxy_geometry_brick.cpp $(ADDITINAL_INCLUDE_DIR)		

#Entry exit points(MPR)
mi_mpr_entry_exit_points.o: $(EXPORT_HEADER) mi_mpr_entry_exit_points.h mi_mpr_entry_exit_points.cpp mi_ray_caster_define.h \
mi_camera_calculator.o mi_entry_exit_points.o
	$(CC) -c -fPIC $(CFLAGS) -Wno-unused-variable mi_mpr_entry_exit_points.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_entry_exit_points.o: $(EXPORT_HEADER) mi_entry_exit_points.h mi_entry_exit_points.cpp mi_camera_calculator.o
	$(CC) -c -fPIC $(CFLAGS) mi_entry_exit_points.cpp $(ADDITINAL_INCLUDE_DIR)	

#Transfer function
mi_transfer_function_loader.o: $(EXPORT_HEADER) mi_transfer_function_loader.h mi_transfer_function_loader.cpp mi_ray_caster_define.h mi_color_transfer_function.o mi_opacity_transfer_function.o
	$(CC) -c -fPIC $(CFLAGS) mi_transfer_function_loader.cpp $(ADDITINAL_INCLUDE_DIR)

mi_color_transfer_function.o: $(EXPORT_HEADER) mi_color_transfer_function.h mi_color_transfer_function.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_color_transfer_function.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_opacity_transfer_function.o: $(EXPORT_HEADER) mi_opacity_transfer_function.h mi_opacity_transfer_function.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_opacity_transfer_function.cpp $(ADDITINAL_INCLUDE_DIR)	

#Camera related
mi_camera_interactor.o: $(EXPORT_HEADER) mi_camera_interactor.h mi_camera_interactor.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_camera_interactor.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_camera_calculator.o: $(EXPORT_HEADER) mi_camera_calculator.h mi_camera_calculator.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_camera_calculator.cpp $(ADDITINAL_INCLUDE_DIR)	

#Brick related
mi_brick_pool.o: $(EXPORT_HEADER) mi_brick_pool.h mi_brick_pool.cpp mi_brick_define.h mi_brick_info_calculator.o	
	$(CC) -c -fPIC $(CFLAGS) mi_brick_pool.cpp $(ADDITINAL_INCLUDE_DIR)

mi_brick_info_calculator.o: $(EXPORT_HEADER) mi_brick_info_calculator.h mi_brick_info_calculator.cpp mi_brick_define.h
	$(CC) -c -fPIC $(CFLAGS) -Wno-unused-variable mi_brick_info_calculator.cpp $(ADDITINAL_INCLUDE_DIR)

mi_mask_label_store.o: $(EXPORT_HEADER) mi_mask_label_store.h mi_mask_label_store.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_mask_label_store.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_graphic_object_navigator.o: $(EXPORT_HEADER) mi_graphic_object_navigator.h mi_graphic_object_navigator.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_graphic_object_navigator.cpp $(ADDITINAL_INCLUDE_DIR)	


.PHONY : clean
clean:
	-rm *.o $(TARGET)
