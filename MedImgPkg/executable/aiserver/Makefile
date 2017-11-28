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

LOCAL_LIB_DIR = /usr/local/lib

VPATH = $(INCLUDE_ROOT_DIR)

OBJS = mi_ai_server_controller.o mi_ai_server_thread_model.o  mi_ai_cmd_handler_operating.o mi_ai_lung_evaluate_py_wrapper.o \
mi_ai_operation_inference.o 

TARGET = $(BIN_DIR)/aiserver

ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR)
ADDITIANL_DEPENDENCIES = -L$(LOCAL_LIB_DIR) -L$(BIN_DIR) -lpthread -lboost_system -lboost_filesystem -lboost_thread -lboost_log -lboost_log_setup -lprotobuf -lpython2.7 \
 -lmedimg_log -lmedimg_util -lmedimg_glresource -lmedimg_arithmetic -lmedimg_io -lmedimg_renderalgo -lmedimg_appcommon

EXPORT_HEADER = mi_ai_server_common.h

$(TARGET): mi_ai_server_main.o $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) mi_ai_server_main.o $(OBJS) $(ADDITIANL_DEPENDENCIES) 

mi_ai_server_main.o: $(EXPORT_HEADER)  mi_ai_server_main.cpp mi_ai_server_controller.o
	$(CC) -c -fPIC $(CFLAGS) mi_ai_server_main.cpp $(ADDITINAL_INCLUDE_DIR)

mi_ai_server_controller.o : mi_ai_server_controller.cpp mi_ai_server_controller.h \
mi_ai_server_thread_model.o mi_ai_operation_inference.o mi_ai_cmd_handler_operating.o
	$(CC) -c -fPIC $(CFLAGS) mi_ai_server_controller.cpp $(ADDITINAL_INCLUDE_DIR)

mi_ai_operation_inference.o: mi_ai_operation_inference.cpp mi_ai_operation_inference.h mi_ai_lung_evaluate_py_wrapper.o
	$(CC) -c -fPIC $(CFLAGS) mi_ai_operation_inference.cpp $(ADDITINAL_INCLUDE_DIR)

mi_ai_cmd_handler_operating.o: mi_ai_cmd_handler_operating.cpp mi_ai_cmd_handler_operating.h 
	$(CC) -c -fPIC $(CFLAGS) mi_ai_cmd_handler_operating.cpp $(ADDITINAL_INCLUDE_DIR)

mi_ai_server_thread_model.o : mi_ai_server_thread_model.cpp mi_ai_server_thread_model.h
	$(CC) -c -fPIC $(CFLAGS) mi_ai_server_thread_model.cpp $(ADDITINAL_INCLUDE_DIR)

mi_ai_lung_evaluate_py_wrapper.o: mi_ai_lung_evaluate_py_wrapper.cpp mi_ai_lung_evaluate_py_wrapper.h 
	$(CC) -c -fPIC $(CFLAGS) mi_ai_lung_evaluate_py_wrapper.cpp $(ADDITINAL_INCLUDE_DIR)

.PHONY : clean
clean:
	-rm *.o $(TARGET)

