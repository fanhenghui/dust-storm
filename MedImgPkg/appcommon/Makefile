ver = debug
CC  = g++ -std=c++11 -Wall 

ifeq ($(ver) , release)
else
CFLAGS += -g
endif

ROOT = ..
INCLUDE_ROOT_DIR = $(ROOT)/include
INCLUDE_DIR = $(INCLUDE_ROOT_DIR)/appcommon
BIN_DIR = $(ROOT)/bin

VPATH = $(INCLUDE_ROOT_DIR) : $(INCLUDE_DIR)

OBJS = mi_app_controller.o mi_app_thread_model.o mi_operation_command_handler.o mi_operation_factory.o mi_app_cell.o \
mi_ready_command_handler.o mi_shut_down_command_handler.o mi_app_data_base.o
TARGET = $(BIN_DIR)/libmedimg_appcommon.so

EXPORT_HEADER = mi_app_common_export.h mi_app_common_define.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) -I$(INCLUDE_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -lmedimg_util -lmedimg_glresource -lmedimg_renderalgo -lboost_system -lboost_filesystem -lboost_thread -lmysqlcppconn

$(TARGET): $(OBJS) $(BIN_DIR)/libmedimg_util.so $(BIN_DIR)/libmedimg_glresource.so
	$(CC) -shared $(CFLAGS) -o $(TARGET) $(OBJS) $(ADDITIANL_DEPENDENCIES)

mi_app_controller.o: $(EXPORT_HEADER) mi_app_controller.h mi_app_controller.cpp mi_app_thread_model.o
	$(CC) -c -fPIC $(CFLAGS) mi_app_controller.cpp $(ADDITINAL_INCLUDE_DIR)

mi_operation_command_handler.o: $(EXPORT_HEADER) mi_operation_interface.h mi_operation_command_handler.h mi_operation_command_handler.cpp mi_operation_factory.o mi_app_thread_model.o
	$(CC) -c -fPIC $(CFLAGS) mi_operation_command_handler.cpp $(ADDITINAL_INCLUDE_DIR)

mi_ready_command_handler.o: $(EXPORT_HEADER) mi_operation_interface.h mi_ready_command_handler.h mi_ready_command_handler.cpp mi_app_thread_model.o mi_app_controller.o
	$(CC) -c -fPIC $(CFLAGS) mi_ready_command_handler.cpp $(ADDITINAL_INCLUDE_DIR)

mi_shut_down_command_handler.o: $(EXPORT_HEADER) mi_operation_interface.h mi_shut_down_command_handler.h mi_shut_down_command_handler.cpp mi_app_thread_model.o mi_app_controller.o
	$(CC) -c -fPIC $(CFLAGS) mi_shut_down_command_handler.cpp $(ADDITINAL_INCLUDE_DIR)

mi_app_thread_model.o: $(EXPORT_HEADER) mi_operation_interface.h mi_app_thread_model.h mi_app_thread_model.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_app_thread_model.cpp $(ADDITINAL_INCLUDE_DIR)

mi_operation_factory.o: $(EXPORT_HEADER) mi_operation_interface.h mi_operation_factory.h mi_operation_factory.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_operation_factory.cpp $(ADDITINAL_INCLUDE_DIR)

mi_app_cell.o: $(EXPORT_HEADER) mi_app_cell.h mi_app_cell.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_app_cell.cpp $(ADDITINAL_INCLUDE_DIR)	

mi_app_data_base.o : $(EXPORT_HEADER) mi_app_data_base.h mi_app_data_base.cpp
	$(CC) -c -fPIC $(CFLAGS) mi_app_data_base.cpp $(ADDITINAL_INCLUDE_DIR)	

.PHONY : clean
clean:
	-rm $(OBJS) $(TARGET)
