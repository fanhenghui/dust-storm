ver = debug
CC  = g++ -std=c++11 -Wall 

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

PROTOCO_LIB_DIR = /usr/local/lib
GOOGLE_INCLUDE_DIR = /usr/local/include/google/
PROTOCO_INCLUDE_DIR = /usr/local/include/google/protobuf

VPATH = $(INCLUDE_ROOT_DIR)

OBJS = mi_review_config.o mi_review_controller.o \
mi_model_annotation.o mi_ob_annotation_segment.o mi_ob_annotation_statistic.o \
mi_operation_init.o mi_operation_annotation.o \
mi_cmd_handler_search_worklist.o

TARGET = $(BIN_DIR)/review_server

COMMON_HEADER = mi_review_common.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) -I$(GOOGLE_INCLUDE_DIR) -I$(PROTOCO_INCLUDE_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -L$(PROTOCO_LIB_DIR) -lmedimg_log -lmedimg_util -lmedimg_glresource -lmedimg_arithmetic -lmedimg_io -lmedimg_appcommon -lmedimg_renderalgo \
-lpthread -lboost_system -lboost_filesystem -lboost_thread -lboost_log -lboost_log_setup -lGL -lGLEW -lprotobuf

all: pre-build main-build

pre-build:
	sh protoc-gen.sh

main-build: $(TARGET)

EXPORT_HEADER = mi_review_common.h

$(TARGET): mi_message.pb.o mi_review_main.o $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) mi_message.pb.o mi_review_main.o $(OBJS) $(ADDITIANL_DEPENDENCIES)

mi_review_main.o: $(EXPORT_HEADER)  mi_review_main.cpp  mi_review_controller.o
	$(CC) -c -fPIC $(CFLAGS) mi_review_main.cpp $(ADDITINAL_INCLUDE_DIR)

mi_message.pb.o: mi_message.pb.h mi_message.pb.cc 
	$(CC) -c -fPIC $(CFLAGS) mi_message.pb.cc $(ADDITINAL_INCLUDE_DIR)

$(OBJS):%.o:%.cpp %.h $(COMMON_HEADER)
	$(CC) -c -fPIC $(CFLAGS) $< -o $@ $(ADDITINAL_INCLUDE_DIR)

.PHONY : clean
clean:
	-rm *.o $(TARGET)
	-rm *.pb.cc *.pb.h

