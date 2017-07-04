ver = release
CC  = g++ -std=c++11 -Wall 

ifeq ($(ver) , release)
CFLAGS += -c -fPIC
else
CFLAGS += -c -g -fPIC
endif

ROOT = ..
INCLUDE_ROOT_DIR = $(ROOT)/Include
INCLUDE_DIR = $(INCLUDE_ROOT_DIR)/MedImgAppCommon
BIN_DIR = $(ROOT)/bin

VPATH = $(INCLUDE_ROOT_DIR) : $(INCLUDE_DIR)

OBJS = mi_app_controller.o mi_app_thread_model.o mi_operation_command_handler.o mi_operation_factory.o
TARGET = $(BIN_DIR)/libmedimg_appcommon.so

EXPORT_HEADER = mi_app_common_export.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) -I$(INCLUDE_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -lmedimg_util -lmedimg_glresource -lboost_system -lboost_filesystem -lboost_thread 

$(TARGET): $(OBJS)
	$(CC) -shared -o $(TARGET) $(OBJS) $(ADDITIANL_DEPENDENCIES)

mi_app_controller.o: $(EXPORT_HEADER) mi_app_controller.h mi_app_thread_model.o
	$(CC) $(CFLAGS) mi_app_controller.cpp $(ADDITINAL_INCLUDE_DIR)

mi_operation_command_handler.o: $(EXPORT_HEADER) mi_operation_interface.h mi_operation_command_handler.h mi_operation_command_handler.cpp mi_operation_factory.o mi_app_thread_model.o
	$(CC) $(CFLAGS) mi_operation_command_handler.cpp $(ADDITINAL_INCLUDE_DIR)

mi_app_thread_model.o: $(EXPORT_HEADER) mi_operation_interface.h mi_app_thread_model.h mi_app_thread_model.cpp
	$(CC) $(CFLAGS) mi_app_thread_model.cpp $(ADDITINAL_INCLUDE_DIR)

mi_operation_factory.o: $(EXPORT_HEADER) mi_operation_interface.h mi_operation_factory.h mi_operation_factory.cpp
	$(CC) $(CFLAGS) mi_operation_factory.cpp $(ADDITINAL_INCLUDE_DIR)

.PHONY : clean
clean:
	-rm $(OBJS) $(TARGET)
