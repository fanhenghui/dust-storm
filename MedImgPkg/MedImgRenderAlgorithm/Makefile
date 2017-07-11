ver = debug
CC  = g++ -std=c++11 

ifeq ($(ver) , release)
else
CFLAGS += -g
endif

ROOT = ..
INCLUDE_ROOT_DIR = $(ROOT)/Include
INCLUDE_DIR = $(INCLUDE_ROOT_DIR)/MedImgRenderAlgorithm
BIN_DIR = $(ROOT)/bin

VPATH = $(INCLUDE_ROOT_DIR) : $(INCLUDE_DIR)

OBJS = mi_brick_define.o mi_brick_generator.o mi_brick_info_generator.o mi_brick_utils.o mi_brick_pool.o mi_mask_label_store.o \
mi_camera_calculator.o  mi_camera_interactor.o \
mi_color_transfer_function.o mi_opacity_transfer_function.o \
mi_entry_exit_points.o mi_mpr_entry_exit_points.o \
mi_ray_caster_canvas.o mi_ray_caster_inner_buffer.o mi_ray_caster.o mi_ray_casting_cpu_brick_acc.o mi_ray_casting_cpu.o mi_ray_casting_gpu.o \
mi_rc_step_color_inverse.o mi_rc_step_composite.o mi_rc_step_main.o mi_rc_step_mask_overlay.o \
mi_rc_step_mask_sampler.o mi_rc_step_ray_casting.o mi_rc_step_shading.o mi_rc_step_utils.o mi_rc_step_volume_sampler.o \
mi_volume_infos.o mi_scene_base.o mi_mpr_scene.o mi_ray_cast_scene.o

TARGET = $(BIN_DIR)/libmedimg_renderalgorithm.so

EXPORT_HEADER = mi_render_algo_export.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) -I$(INCLUDE_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -lmedimg_util -lmedimg_arithmetic -lmedimg_glresource -lmedimg_io -lboost_system -lboost_thread -lgpujpeg


all: pre-build main-build

pre-build:
	../bin/medimg_shaderencoder ./Shader/shader_config

main-build: $(TARGET)

$(TARGET): $(OBJS) $(BIN_DIR)/libmedimg_util.so $(BIN_DIR)/libmedimg_glresource.so $(BIN_DIR)/libmedimg_io.so
	$(CC) -shared $(CFLAGS) -o $(TARGET) $(OBJS) $(ADDITIANL_DEPENDENCIES)

#scene related
mi_mpr_scene.o: $(EXPORT_HEADER) mi_mpr_scene.h mi_mpr_scene.cpp mi_scene_base.o \
mi_camera_calculator.o mi_camera_interactor.o mi_volume_infos.o mi_mpr_entry_exit_points.o mi_ray_caster.o
	$(CC) -c -fPIC $(CFLAGS) mi_mpr_scene.cpp $(ADDITINAL_INCLUDE_DIR)

mi_ray_cast_scene.o: $(EXPORT_HEADER) mi_ray_cast_scene.h mi_ray_cast_scene.cpp mi_scene_base.o \
mi_camera_calculator.o mi_camera_interactor.o mi_volume_infos.o mi_entry_exit_points.o mi_ray_caster.o mi_ray_caster_canvas.o
	$(CC) -c -fPIC $(CFLAGS) mi_ray_cast_scene.cpp $(ADDITINAL_INCLUDE_DIR)

mi_scene_base.o: $(EXPORT_HEADER) mi_scene_base.h mi_scene_base.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_scene_base.cpp $(ADDITINAL_INCLUDE_DIR)

mi_volume_infos.o: $(EXPORT_HEADER) mi_volume_infos.h mi_volume_infos.cpp \
mi_camera_calculator.o mi_brick_generator.o mi_brick_info_generator.o mi_brick_pool.o mi_brick_utils.o
	$(CC) -c -fPIC $(CFLAGS) mi_volume_infos.cpp $(ADDITINAL_INCLUDE_DIR)

#Ray casting
mi_ray_caster.o: $(EXPORT_HEADER) mi_ray_caster.h mi_ray_caster.cpp \
mi_ray_caster_canvas.o mi_ray_casting_cpu.o mi_ray_casting_cpu_brick_acc.o mi_ray_caster_inner_buffer.o mi_ray_casting_gpu.o
	$(CC) -c -fPIC $(CFLAGS) mi_ray_caster.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_ray_casting_gpu.o: $(EXPORT_HEADER) mi_ray_casting_gpu.h mi_ray_casting_gpu.cpp \
mi_entry_exit_points.o mi_ray_caster_canvas.o \
mi_rc_step_color_inverse.o mi_rc_step_composite.o mi_rc_step_main.o mi_rc_step_mask_overlay.o \
mi_rc_step_mask_sampler.o mi_rc_step_ray_casting.o mi_rc_step_shading.o mi_rc_step_utils.o mi_rc_step_volume_sampler.o
	$(CC) -c -fPIC $(CFLAGS) mi_ray_casting_gpu.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_ray_casting_cpu.o: $(EXPORT_HEADER) mi_ray_casting_cpu.h mi_ray_casting_cpu.cpp \
mi_entry_exit_points.o mi_ray_caster_canvas.o
	$(CC) -c -fPIC $(CFLAGS) mi_ray_casting_cpu.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_ray_casting_cpu_brick_acc.o: $(EXPORT_HEADER) mi_ray_casting_cpu_brick_acc.h mi_ray_casting_cpu_brick_acc.cpp \
mi_entry_exit_points.o mi_mpr_entry_exit_points.o mi_ray_caster_canvas.o mi_brick_utils.o
	$(CC) -c -fPIC $(CFLAGS) mi_ray_casting_cpu_brick_acc.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_ray_caster_inner_buffer.o: $(EXPORT_HEADER) mi_ray_caster_inner_buffer.h mi_ray_caster_inner_buffer.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_ray_caster_inner_buffer.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_ray_caster_canvas.o: $(EXPORT_HEADER) mi_ray_caster_canvas.h mi_ray_caster_canvas.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_ray_caster_canvas.cpp $(ADDITINAL_INCLUDE_DIR)

#GPU render step
mi_rc_step_color_inverse.o: $(EXPORT_HEADER) mi_rc_step_base.h mi_rc_step_color_inverse.h mi_rc_step_color_inverse.cpp 
	$(CC) -c -fPIC $(CFLAGS) mi_rc_step_color_inverse.cpp $(ADDITINAL_INCLUDE_DIR)

mi_rc_step_composite.o: $(EXPORT_HEADER) mi_rc_step_base.h mi_rc_step_composite.h mi_rc_step_composite.cpp 
	$(CC) -c -fPIC $(CFLAGS) mi_rc_step_composite.cpp $(ADDITINAL_INCLUDE_DIR)

mi_rc_step_main.o: $(EXPORT_HEADER) mi_rc_step_base.h mi_rc_step_main.h mi_rc_step_main.cpp 
	$(CC) -c -fPIC $(CFLAGS) mi_rc_step_main.cpp $(ADDITINAL_INCLUDE_DIR)

mi_rc_step_mask_overlay.o: $(EXPORT_HEADER) mi_rc_step_base.h mi_rc_step_mask_overlay.h mi_rc_step_mask_overlay.cpp \
mi_ray_caster_inner_buffer.o
	$(CC) -c -fPIC $(CFLAGS) mi_rc_step_mask_overlay.cpp $(ADDITINAL_INCLUDE_DIR)

mi_rc_step_mask_sampler.o: $(EXPORT_HEADER) mi_rc_step_base.h mi_rc_step_mask_sampler.h mi_rc_step_mask_sampler.cpp 
	$(CC) -c -fPIC $(CFLAGS) mi_rc_step_mask_sampler.cpp $(ADDITINAL_INCLUDE_DIR)

mi_rc_step_ray_casting.o: $(EXPORT_HEADER) mi_rc_step_base.h mi_rc_step_ray_casting.h mi_rc_step_ray_casting.cpp 
	$(CC) -c -fPIC $(CFLAGS) mi_rc_step_ray_casting.cpp $(ADDITINAL_INCLUDE_DIR)

mi_rc_step_shading.o: $(EXPORT_HEADER) mi_rc_step_base.h mi_rc_step_shading.h mi_rc_step_shading.cpp 
	$(CC) -c -fPIC $(CFLAGS) mi_rc_step_shading.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_rc_step_utils.o: $(EXPORT_HEADER) mi_rc_step_base.h mi_rc_step_utils.h mi_rc_step_utils.cpp 
	$(CC) -c -fPIC $(CFLAGS) mi_rc_step_utils.cpp $(ADDITINAL_INCLUDE_DIR)

mi_rc_step_volume_sampler.o: $(EXPORT_HEADER) mi_rc_step_base.h mi_rc_step_volume_sampler.h mi_rc_step_volume_sampler.cpp 
	$(CC) -c -fPIC $(CFLAGS) mi_rc_step_volume_sampler.cpp $(ADDITINAL_INCLUDE_DIR)			

#Entry exit points(MPR)
mi_mpr_entry_exit_points.o: $(EXPORT_HEADER) mi_mpr_entry_exit_points.h mi_mpr_entry_exit_points.cpp mi_ray_caster_define.h \
mi_camera_calculator.o mi_entry_exit_points.o
	$(CC) -c -fPIC $(CFLAGS) mi_mpr_entry_exit_points.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_entry_exit_points.o: $(EXPORT_HEADER) mi_entry_exit_points.h mi_entry_exit_points.cpp mi_camera_calculator.o
	$(CC) -c -fPIC $(CFLAGS) mi_entry_exit_points.cpp $(ADDITINAL_INCLUDE_DIR)	

#Transfer function
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
mi_brick_pool.o: $(EXPORT_HEADER) mi_brick_pool.h mi_brick_pool.cpp \
mi_brick_define.o mi_brick_utils.o mi_brick_generator.o mi_brick_info_generator.o
	$(CC) -c -fPIC $(CFLAGS) mi_brick_pool.cpp $(ADDITINAL_INCLUDE_DIR)

mi_brick_info_generator.o: $(EXPORT_HEADER) mi_brick_info_generator.h mi_brick_info_generator.cpp mi_brick_define.o
	$(CC) -c -fPIC $(CFLAGS) mi_brick_info_generator.cpp $(ADDITINAL_INCLUDE_DIR)

mi_brick_generator.o: $(EXPORT_HEADER) mi_brick_generator.h mi_brick_generator.cpp mi_brick_utils.o mi_brick_define.o
	$(CC) -c -fPIC $(CFLAGS) mi_brick_generator.cpp $(ADDITINAL_INCLUDE_DIR)

mi_brick_define.o: $(EXPORT_HEADER) mi_brick_define.h mi_brick_define.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_brick_define.cpp $(ADDITINAL_INCLUDE_DIR)

mi_brick_utils.o: $(EXPORT_HEADER) mi_brick_utils.h mi_brick_utils.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_brick_utils.cpp $(ADDITINAL_INCLUDE_DIR)

mi_mask_label_store.o: $(EXPORT_HEADER) mi_mask_label_store.h mi_mask_label_store.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_mask_label_store.cpp $(ADDITINAL_INCLUDE_DIR)	

.PHONY : clean
clean:
	-rm $(OBJS) $(TARGET)
