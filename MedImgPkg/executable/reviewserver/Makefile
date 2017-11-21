ver = debug
CC  = g++ -std=c++11 -Wall -Wno-unused-function

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

OBJS = mi_review_controller.o mi_operation_init.o \

TARGET = $(BIN_DIR)/review_server

COMMON_HEADER = mi_review_common.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -L$(LOCAL_LIB_DIR) -lpthread -lboost_system -lboost_filesystem -lboost_thread -lboost_log -lboost_log_setup -lprotobuf\
-lmedimg_log -lmedimg_util -lmedimg_glresource -lmedimg_arithmetic -lmedimg_io -lmedimg_renderalgo -lmedimg_appcommon

EXPORT_HEADER = mi_review_common.h

$(TARGET): mi_review_main.o $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) mi_review_main.o $(OBJS) $(ADDITIANL_DEPENDENCIES)

mi_review_main.o: $(EXPORT_HEADER)  mi_review_main.cpp  mi_review_controller.o
	$(CC) -c -fPIC $(CFLAGS) mi_review_main.cpp $(ADDITINAL_INCLUDE_DIR)

$(OBJS):%.o:%.cpp %.h $(COMMON_HEADER)
	$(CC) -c -fPIC $(CFLAGS) $< -o $@ $(ADDITINAL_INCLUDE_DIR)

.PHONY : clean
clean:
	-rm *.o $(TARGET)
