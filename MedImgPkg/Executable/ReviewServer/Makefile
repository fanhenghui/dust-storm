ver = debug
CC  = g++ -std=c++11 -Wall 
DEFS = -D VR -D LOAD_LOCAL

ifeq ($(ver) , release)
else
CFLAGS += -g $(DEFS)
endif

ROOT = ../..
INCLUDE_ROOT_DIR = $(ROOT)/Include
BIN_DIR = $(ROOT)/bin

PROTOCO_LIB_DIR = /usr/local/protobuf/lib
GOOGLE_INCLUDE_DIR = /usr/local/protobuf/include/google/
PROTOCO_INCLUDE_DIR = /usr/local/protobuf/include/

VPATH = $(INCLUDE_ROOT_DIR)

OBJS = mi_review_controller.o mi_load_series_command_handler.o mi_operation_mpr_paging.o mi_review_main.o mi_mpr_play_command_handler.o \
mi_review_config.o mi_message.pb.o mi_operation_init.o
TARGET = $(BIN_DIR)/review_server

COMMON_HEADER = mi_review_common.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) -I$(GOOGLE_INCLUDE_DIR) -I$(PROTOCO_INCLUDE_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -L$(PROTOCO_LIB_DIR) -lmedimg_util -lmedimg_glresource -lmedimg_arithmetic -lmedimg_io -lmedimg_appcommon -lmedimg_renderalgorithm \
-lboost_system -lboost_filesystem -lboost_thread -lGL -lGLEW -lprotobuf

$(TARGET): $(OBJS) $(BIN_DIR)/libmedimg_util.so $(BIN_DIR)/libmedimg_glresource.so $(BIN_DIR)/libmedimg_io.so $(BIN_DIR)/libmedimg_appcommon.so $(BIN_DIR)/libmedimg_renderalgorithm.so
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(ADDITIANL_DEPENDENCIES) $(ADDITINAL_INCLUDE_DIR)

EXPORT_HEADER = mi_review_common.h

mi_review_main.o: $(EXPORT_HEADER)  mi_review_main.cpp  mi_review_controller.o
	$(CC) -c -fPIC $(CFLAGS) mi_review_main.cpp $(ADDITINAL_INCLUDE_DIR)

mi_load_series_command_handler.o: $(EXPORT_HEADER) mi_load_series_command_handler.h mi_load_series_command_handler.cpp \
mi_review_controller.o mi_review_config.o mi_message.pb.o
	$(CC) -c -fPIC $(CFLAGS) mi_load_series_command_handler.cpp $(ADDITINAL_INCLUDE_DIR)

mi_mpr_play_command_handler.o: $(EXPORT_HEADER) mi_mpr_play_command_handler.h mi_mpr_play_command_handler.cpp \
mi_review_controller.o mi_message.pb.o
	$(CC) -c -fPIC $(CFLAGS) mi_mpr_play_command_handler.cpp $(ADDITINAL_INCLUDE_DIR)

mi_operation_init.o: $(EXPORT_HEADER) mi_operation_init.h mi_operation_init.cpp \
mi_review_controller.o mi_message.pb.o
	$(CC) -c -fPIC $(CFLAGS) mi_operation_init.cpp $(ADDITINAL_INCLUDE_DIR)

mi_operation_mpr_paging.o: $(EXPORT_HEADER) mi_operation_mpr_paging.h mi_operation_mpr_paging.cpp \
mi_review_controller.o mi_message.pb.o
	$(CC) -c -fPIC $(CFLAGS) mi_operation_mpr_paging.cpp $(ADDITINAL_INCLUDE_DIR)

mi_review_controller.o: $(EXPORT_HEADER) mi_review_controller.h mi_review_controller.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_review_controller.cpp $(ADDITINAL_INCLUDE_DIR)

mi_review_config.o: $(EXPORT_HEADER) mi_review_config.h mi_review_config.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_review_config.cpp $(ADDITINAL_INCLUDE_DIR)

mi_message.pb.o: mi_message.pb.h mi_message.pb.cc 
	$(CC) -c -fPIC $(CFLAGS) mi_message.pb.cc $(ADDITINAL_INCLUDE_DIR)

.PHONY : clean
clean:
	-rm $(OBJS) $(TARGET)

