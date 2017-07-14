ver = debug
CC  = g++ -std=c++11 -Wall 

ifeq ($(ver) , release)
else
CFLAGS += -g
endif

ROOT = ../..
INCLUDE_ROOT_DIR = $(ROOT)/Include
BIN_DIR = $(ROOT)/bin

VPATH = $(INCLUDE_ROOT_DIR)


OBJS = mi_review_controller.o mi_load_series_command_handler.o mi_operation_mpr_paging.o main.o mi_mpr_play_command_handler.o
TARGET = $(BIN_DIR)/review_server

COMMON_HEADER = mi_review_common.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) 
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -lmedimg_util -lmedimg_glresource -lmedimg_arithmetic -lmedimg_io -lmedimg_appcommon -lmedimg_renderalgorithm -lboost_system -lboost_filesystem -lboost_thread -lGL -lGLEW 

$(TARGET): $(OBJS) $(BIN_DIR)/libmedimg_util.so $(BIN_DIR)/libmedimg_glresource.so $(BIN_DIR)/libmedimg_io.so $(BIN_DIR)/libmedimg_appcommon.so $(BIN_DIR)/libmedimg_renderalgorithm.so
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(ADDITIANL_DEPENDENCIES) $(ADDITINAL_INCLUDE_DIR)

main.o: $(EXPORT_HEADER)  main.cpp  mi_load_series_command_handler.o mi_review_controller.o
	$(CC) -c -fPIC $(CFLAGS) main.cpp $(ADDITINAL_INCLUDE_DIR)

mi_load_series_command_handler.o: $(EXPORT_HEADER) mi_load_series_command_handler.h mi_load_series_command_handler.cpp \
mi_review_controller.o
	$(CC) -c -fPIC $(CFLAGS) mi_load_series_command_handler.cpp $(ADDITINAL_INCLUDE_DIR)

mi_mpr_play_command_handler.o: $(EXPORT_HEADER) mi_mpr_play_command_handler.h mi_mpr_play_command_handler.cpp \
mi_review_controller.o
	$(CC) -c -fPIC $(CFLAGS) mi_mpr_play_command_handler.cpp $(ADDITINAL_INCLUDE_DIR)

mi_operation_mpr_paging.o: $(EXPORT_HEADER) mi_operation_mpr_paging.h mi_operation_mpr_paging.cpp \
mi_review_controller.o
	$(CC) -c -fPIC $(CFLAGS) mi_operation_mpr_paging.cpp $(ADDITINAL_INCLUDE_DIR)

mi_review_controller.o: $(EXPORT_HEADER) mi_review_controller.h mi_review_controller.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_review_controller.cpp $(ADDITINAL_INCLUDE_DIR)

.PHONY : clean
clean:
	-rm $(OBJS) $(TARGET)

