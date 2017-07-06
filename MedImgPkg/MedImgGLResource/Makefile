ver = debug
CC  = g++ -std=c++11 -Wall 

ifeq ($(ver) , release)
else
CFLAGS += -g
endif

ROOT = ..
INCLUDE_ROOT_DIR = $(ROOT)/Include
INCLUDE_DIR = $(INCLUDE_ROOT_DIR)/MedImgGLResource
BIN_DIR = $(ROOT)/bin

VPATH = $(INCLUDE_ROOT_DIR) : $(INCLUDE_DIR)

OBJS = mi_gl_buffer.o mi_gl_context.o mi_gl_environment.o mi_gl_fbo.o mi_gl_program.o  \
mi_gl_resource_manager_container.o mi_gl_texture_1d_array.o mi_gl_texture_1d.o mi_gl_texture_2d.o mi_gl_texture_3d.o \
mi_gl_texture_base.o mi_gl_utils.o mi_gl_vao.o
TARGET = $(BIN_DIR)/libmedimg_glresource.so

EXPORT_HEADER = mi_gl_resource_export.h mi_gl_object.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) -I$(INCLUDE_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -lmedimg_util -lGL -lGLEW

$(TARGET): $(OBJS)
	$(CC) -shared $(CFLAGS) -o $(TARGET) $(OBJS) $(ADDITIANL_DEPENDENCIES)

mi_gl_resource_manager_container.o: $(EXPORT_HEADER) mi_gl_resource_manager_container.h mi_gl_resource_manager_container.cpp \
mi_gl_resource_manager.h mi_gl_resource_manager.inl 
	$(CC) -c -fPIC $(CFLAGS) mi_gl_resource_manager_container.cpp $(ADDITINAL_INCLUDE_DIR)

mi_gl_object.o: $(EXPORT_HEADER) mi_gl_object.h mi_gl_object.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_gl_object.cpp $(ADDITINAL_INCLUDE_DIR)

mi_gl_buffer.o: $(EXPORT_HEADER) mi_gl_buffer.h mi_gl_buffer.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_gl_buffer.cpp $(ADDITINAL_INCLUDE_DIR)

mi_gl_context.o: $(EXPORT_HEADER) mi_gl_context.h mi_gl_context.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_gl_context.cpp $(ADDITINAL_INCLUDE_DIR)

mi_gl_environment.o: $(EXPORT_HEADER) mi_gl_environment.h mi_gl_environment.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_gl_environment.cpp $(ADDITINAL_INCLUDE_DIR)

mi_gl_fbo.o: $(EXPORT_HEADER) mi_gl_fbo.h mi_gl_fbo.cpp mi_gl_utils.o
	$(CC) -c -fPIC $(CFLAGS) mi_gl_fbo.cpp $(ADDITINAL_INCLUDE_DIR)

mi_gl_vao.o: $(EXPORT_HEADER) mi_gl_vao.h mi_gl_vao.cpp mi_gl_texture_2d.o
	$(CC) -c -fPIC $(CFLAGS) mi_gl_vao.cpp $(ADDITINAL_INCLUDE_DIR)

mi_gl_program.o: $(EXPORT_HEADER) mi_gl_program.h mi_gl_program.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_gl_program.cpp $(ADDITINAL_INCLUDE_DIR)

mi_gl_texture_1d.o: $(EXPORT_HEADER) mi_gl_texture_1d.h mi_gl_texture_1d.cpp mi_gl_texture_base.o
	$(CC) -c -fPIC $(CFLAGS) mi_gl_texture_1d.cpp $(ADDITINAL_INCLUDE_DIR)

mi_gl_texture_2d.o: $(EXPORT_HEADER) mi_gl_texture_2d.h mi_gl_texture_2d.cpp mi_gl_texture_base.o
	$(CC) -c -fPIC $(CFLAGS) mi_gl_texture_2d.cpp $(ADDITINAL_INCLUDE_DIR)

mi_gl_texture_3d.o: $(EXPORT_HEADER) mi_gl_texture_3d.h mi_gl_texture_3d.cpp mi_gl_texture_base.o
	$(CC) -c -fPIC $(CFLAGS) mi_gl_texture_3d.cpp $(ADDITINAL_INCLUDE_DIR)

mi_gl_texture_1d_array.o: $(EXPORT_HEADER) mi_gl_texture_1d_array.h mi_gl_texture_1d_array.cpp mi_gl_texture_base.o
	$(CC) -c -fPIC $(CFLAGS) mi_gl_texture_1d_array.cpp $(ADDITINAL_INCLUDE_DIR)

mi_gl_texture_base.o: $(EXPORT_HEADER) mi_gl_texture_base.h mi_gl_texture_base.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_gl_texture_base.cpp $(ADDITINAL_INCLUDE_DIR)

mi_gl_utils.o: $(EXPORT_HEADER) mi_gl_utils.h mi_gl_utils.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_gl_utils.cpp $(ADDITINAL_INCLUDE_DIR)

.PHONY : clean
clean:
	-rm $(OBJS) $(TARGET)
