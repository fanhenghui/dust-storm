ver = release
CC = g++ -std=c++11 -Wall

ifeq ($(ver),release)
CFLAGS += -O3
else 
CFLAGS += -g
endif

DEF = -DBOOST_LOG_DYN_LINK
CFLAGS += $(DEF)

ROOT = ..
INCLUDE_ROOT_DIR = $(ROOT)/include
INCLUDE_DIR = $(INCLUDE_ROOT_DIR)/cudaresource
CUDA_INCLUDE_DIR = $(CUDA_HOME)/include
BIN_DIR = $(ROOT)/bin

LOCAL_LIB_DIR = /usr/local/lib
CUDA_LIB_DIR = $(CUDA_HOME)/lib64
VPATH = $(INCLUDE_ROOT_DIR) : $(INCLUDE_DIR)

CUDA_OBJS = mi_cuda_object.o mi_cuda_global_memory.o mi_cuda_texture_base.o mi_cuda_texture_1d.o mi_cuda_texture_2d.o mi_cuda_texture_3d.o \
mi_cuda_gl_texture_2d.o mi_cuda_texture_1d_array.o mi_cuda_surface_base.o mi_cuda_surface_2d.o mi_cuda_time_query.o 

OBJS = mi_cuda_utils.o $(CUDA_OBJS) mi_cuda_gl_interop_cache.o mi_cuda_resource_manager.o
TARGET = $(BIN_DIR)/libmedimg_cudaresource.so 

ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) -I$(INCLUDE_DIR) -I$(CUDA_INCLUDE_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -L$(LOCAL_LIB_DIR) -lpthread -lboost_system -lboost_filesystem -lboost_thread -lboost_log_setup -L$(CUDA_LIB_DIR) -lcudart \
-lmedimg_log -lmedimg_util -lmedimg_glresource

$(TARGET): $(OBJS)
	$(CC) -shared $(CFLAGS) -o $(TARGET) $(OBJS) $(ADDITIANL_DEPENDENCIES)

mi_cuda_utils.o: mi_cuda_utils.h mi_cuda_utils.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_cuda_utils.cpp $(ADDITINAL_INCLUDE_DIR)

$(CUDA_OBJS):%.o: %.cpp %.h mi_cuda_utils.o
	$(CC) -c -fPIC $(CFLAGS) -o $@ $< $(ADDITINAL_INCLUDE_DIR)
	
mi_cuda_resource_manager.o: mi_cuda_resource_manager.h mi_cuda_resource_manager.cpp $(CUDA_OBJS) 
	$(CC) -c -fPIC $(CFLAGS) mi_cuda_resource_manager.cpp $(ADDITINAL_INCLUDE_DIR)

mi_cuda_gl_interop_cache.o: mi_cuda_gl_interop_cache.h mi_cuda_gl_interop_cache.cpp mi_cuda_gl_texture_2d.o
	$(CC) -c -fPIC $(CFLAGS) mi_cuda_gl_interop_cache.cpp $(ADDITINAL_INCLUDE_DIR)

.PHONY: clean
clean: 
	-rm *.o $(TARGET)



