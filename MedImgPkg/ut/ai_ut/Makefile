ver = release
CC = g++ -std=c++11 -Wall 

ifeq ($(ver) , release) 
CFLAGS += -O3
else
CFLAGS += -g
endif

DEFS = -DBOOST_LOG_DYN_LINK -D_DEBUG
CFLAGS += $(DEFS)

ROOT = ../..
INCLUDE_ROOT_DIR = $(ROOT)/include
BIN_DIR = $(ROOT)/bin

LOCAL_LIB_DIR = /usr/local/lib

VPATH = $(INCLUDE_ROOT_DIR)

TARGET = $(BIN_DIR)/medimg_ai_ut

ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR)
ADDITIANL_DEPENDENCIES = -L$(LOCAL_LIB_DIR) -L$(BIN_DIR) -lpthread -lboost_system -lboost_filesystem -lboost_thread -lboost_log -lboost_log_setup -lprotobuf -lpython2.7 \
 -lmedimg_log -lmedimg_util -lmedimg_glresource -lmedimg_arithmetic -lmedimg_io -lmedimg_renderalgo -lmedimg_appcommon

$(TARGET): mi_ai_lung_evaluate_py_wrapper.o mi_ai_ut.cpp
	$(CC) $(CFLAGS) -o $(TARGET) mi_ai_ut.cpp mi_ai_lung_evaluate_py_wrapper.o $(ADDITIANL_DEPENDENCIES) $(ADDITINAL_INCLUDE_DIR)

mi_ai_lung_evaluate_py_wrapper.o: mi_ai_lung_evaluate_py_wrapper.cpp mi_ai_lung_evaluate_py_wrapper.h 
	$(CC) -c -fPIC $(CFLAGS) mi_ai_lung_evaluate_py_wrapper.cpp $(ADDITINAL_INCLUDE_DIR)

.PHONY : clean
clean:
	-rm *.o $(TARGET)