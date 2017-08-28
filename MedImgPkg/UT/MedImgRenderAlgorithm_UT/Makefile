ver = debug
CC  = g++ -std=c++11 -Wall 

ifeq ($(ver) , release)

else
CFLAGS += -g
endif

ROOT = ../..
INCLUDE_ROOT_DIR = $(ROOT)/include
BIN_DIR = $(ROOT)/bin

VPATH = $(INCLUDE_ROOT_DIR)

TARGET = $(BIN_DIR)/medimg_renderalgo_ut

ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -L/usr/local/cuda-8.0/targets/x86_64-linux/lib/stubs/ -lmedimg_util -lmedimg_arithmetic -lmedimg_glresource \
-lmedimg_io -lmedimg_renderalgo -lGL -lGLEW -lboost_system -lboost_thread -lglut -lgpujpeg

$(TARGET):  mi_render_algo_main.cpp mi_render_algo_mpr_scene.cpp mi_render_algo_vr_scene.cpp $(BIN_DIR)/libmedimg_util.so $(BIN_DIR)/libmedimg_arithmetic.so $(BIN_DIR)/libmedimg_glresource.so $(BIN_DIR)/libmedimg_io.so $(BIN_DIR)/libmedimg_renderalgo.so
	$(CC) $(CFLAGS) $(ADDITINAL_INCLUDE_DIR) -o $(TARGET)  mi_render_algo_main.cpp mi_render_algo_mpr_scene.cpp mi_render_algo_vr_scene.cpp $(ADDITIANL_DEPENDENCIES)

.PHONY : clean
clean:
	-rm $(TARGET)
