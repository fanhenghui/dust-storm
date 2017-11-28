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

OPERATION_OBJS =  mi_db_operation_query_end.o mi_db_operation_query_preprocess_mask.o \
mi_db_operation_query_ai_annotation.o mi_db_operation_query_dicom.o mi_db_operation_request_evaluation.o \
mi_db_operation_receive_evaluation.o mi_db_operation_receive_ai_ready.o

OBJS = mi_db_server_controller.o mi_db_server_thread_model.o mi_db_cmd_handler_be_operating.o mi_db_cmd_handler_ais_operating.o $(OPERATION_OBJS)

TARGET = $(BIN_DIR)/dbserver

COMMON_HEADER = mi_db_server_common.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -L$(LOCAL_LIB_DIR) -lpthread -lboost_system -lboost_filesystem -lboost_thread -lboost_log -lboost_log_setup -lprotobuf \
-lmedimg_log -lmedimg_util -lmedimg_glresource -lmedimg_arithmetic -lmedimg_io -lmedimg_renderalgo -lmedimg_appcommon

EXPORT_HEADER = mi_db_server_common.h

$(TARGET): mi_db_server_main.o $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) mi_db_server_main.o $(OBJS) $(ADDITIANL_DEPENDENCIES)

mi_db_server_main.o: $(EXPORT_HEADER)  mi_db_server_main.cpp mi_db_server_controller.o
	$(CC) -c -fPIC $(CFLAGS) mi_db_server_main.cpp $(ADDITINAL_INCLUDE_DIR)

mi_db_server_controller.o : mi_db_server_controller.cpp mi_db_server_controller.h mi_db_server_thread_model.o $(OPERATION_OBJS)
	$(CC) -c -fPIC $(CFLAGS) mi_db_server_controller.cpp $(ADDITINAL_INCLUDE_DIR)

mi_db_cmd_handler_be_operating.o: mi_db_cmd_handler_be_operating.cpp mi_db_cmd_handler_be_operating.h mi_db_server_thread_model.o
	$(CC) -c -fPIC $(CFLAGS) mi_db_cmd_handler_be_operating.cpp $(ADDITINAL_INCLUDE_DIR)

mi_db_cmd_handler_ais_operating.o: mi_db_cmd_handler_ais_operating.cpp mi_db_cmd_handler_ais_operating.h mi_db_server_thread_model.o
	$(CC) -c -fPIC $(CFLAGS) mi_db_cmd_handler_ais_operating.cpp $(ADDITINAL_INCLUDE_DIR)

mi_db_server_thread_model.o : mi_db_server_thread_model.cpp mi_db_server_thread_model.h
	$(CC) -c -fPIC $(CFLAGS) mi_db_server_thread_model.cpp $(ADDITINAL_INCLUDE_DIR)

$(OPERATION_OBJS):%.o:%.cpp %.h $(COMMON_HEADER)
	$(CC) -c -fPIC $(CFLAGS) $< -o $@ $(ADDITINAL_INCLUDE_DIR)

.PHONY : clean
clean:
	-rm *.o $(TARGET)

