ver = release
CC  = g++ -std=c++11 -Wall -Wno-unused-function

ifeq ($(ver) , release)
CFLAGS += -O3
else
CFLAGS += -g
endif

DEFS = -DBOOST_LOG_DYN_LINK
CFLAGS += $(DEFS)

ROOT = ../..
INCLUDE_ROOT_DIR = $(ROOT)/include
CUDA_INCLUDE_DIR = $(CUDA_HOME)/include
BIN_DIR = $(ROOT)/bin

LOCAL_LIB_DIR = /usr/local/lib
CUDA_LIB_DIR = $(CUDA_HOME)/lib64

VPATH = $(INCLUDE_ROOT_DIR)

OBJS = mi_render_algo_texture.o mi_render_algo_brick_info.o mi_render_algo_mpr_scene.o mi_render_algo_vr_scene.o 
TARGET = $(BIN_DIR)/medimg_renderalgo_ut

ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) -I$(CUDA_INCLUDE_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -L$(LOCAL_LIB_DIR) -L$(CUDA_LIB_DIR) \
-lGL -lGLEW -lcudart -lpthread -lboost_system -lboost_thread -lboost_log -lboost_log_setup -lglut \
-lmedimg_log -lmedimg_util -lmedimg_arithmetic -lmedimg_glresource -lmedimg_cudaresource -lmedimg_io -lmedimg_renderalgo 

$(TARGET):  mi_render_algo_main.o $(OBJS)
	$(CC) $(CFLAGS) $(ADDITINAL_INCLUDE_DIR) -o $(TARGET) mi_render_algo_main.o $(OBJS) $(ADDITIANL_DEPENDENCIES)

mi_render_algo_main.o: %.o: %.cpp $(OBJS)
	$(CC) -c -fPIC $(CFLAGS) mi_render_algo_main.cpp $(ADDITINAL_INCLUDE_DIR)

$(OBJS): %.o: %.cpp
	$(CC) -c -fPIC $(CFLAGS) -o $@ $< $(ADDITINAL_INCLUDE_DIR)


.PHONY : clean
clean:
	-rm *.o $(TARGET)
