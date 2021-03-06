ver = release
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

OPERATION_OBJS =  mi_db_operation_be_request_end.o mi_db_operation_be_fetch_preprocess_mask.o \
mi_db_operation_be_fetch_ai_evaluation.o mi_db_operation_be_fetch_dicom.o \
mi_db_operation_ai_send_evaluation.o mi_db_operation_ai_ready.o \
mi_db_operation_be_pacs_query.o mi_db_operation_be_pacs_retrieve.o 

OBJS = mi_db_server_controller.o mi_db_server_thread_model.o mi_db_cmd_handler_be_operation.o mi_db_cmd_handler_ai_operation.o \
mi_db_evaluatiion_dispatcher.o mi_db_server_console_echo.o $(OPERATION_OBJS)

TARGET = $(BIN_DIR)/dbserver

COMMON_HEADER = mi_db_server_common.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -L$(LOCAL_LIB_DIR) -lpthread -lboost_system -lboost_filesystem -lboost_thread -lboost_log -lboost_log_setup -lprotobuf \
-lmedimg_log -lmedimg_util -lmedimg_glresource -lmedimg_arithmetic -lmedimg_io

EXPORT_HEADER = mi_db_server_common.h

$(TARGET): mi_db_server_main.o $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) mi_db_server_main.o $(OBJS) $(ADDITIANL_DEPENDENCIES)

mi_db_server_main.o: $(EXPORT_HEADER)  mi_db_server_main.cpp mi_db_server_controller.o
	$(CC) -c -fPIC $(CFLAGS) mi_db_server_main.cpp $(ADDITINAL_INCLUDE_DIR)

mi_db_server_controller.o : mi_db_server_controller.cpp mi_db_server_controller.h mi_db_server_thread_model.o \
mi_db_evaluatiion_dispatcher.o mi_db_server_console_echo.o $(OPERATION_OBJS)
	$(CC) -c -fPIC $(CFLAGS) mi_db_server_controller.cpp $(ADDITINAL_INCLUDE_DIR)

mi_db_cmd_handler_be_operation.o: mi_db_cmd_handler_be_operation.cpp mi_db_cmd_handler_be_operation.h mi_db_server_thread_model.o
	$(CC) -c -fPIC $(CFLAGS) mi_db_cmd_handler_be_operation.cpp $(ADDITINAL_INCLUDE_DIR)

mi_db_cmd_handler_ai_operation.o: mi_db_cmd_handler_ai_operation.cpp mi_db_cmd_handler_ai_operation.h mi_db_server_thread_model.o
	$(CC) -c -fPIC $(CFLAGS) mi_db_cmd_handler_ai_operation.cpp $(ADDITINAL_INCLUDE_DIR)

mi_db_server_thread_model.o : mi_db_server_thread_model.cpp mi_db_server_thread_model.h
	$(CC) -c -fPIC $(CFLAGS) mi_db_server_thread_model.cpp $(ADDITINAL_INCLUDE_DIR)

$(OPERATION_OBJS):%.o:%.cpp %.h $(COMMON_HEADER) mi_db_evaluatiion_dispatcher.o
	$(CC) -c -fPIC $(CFLAGS) $< -o $@ $(ADDITINAL_INCLUDE_DIR)

mi_db_evaluatiion_dispatcher.o : mi_db_evaluatiion_dispatcher.cpp mi_db_evaluatiion_dispatcher.h mi_db_server_thread_model.o
	$(CC) -c -fPIC $(CFLAGS) mi_db_evaluatiion_dispatcher.cpp $(ADDITINAL_INCLUDE_DIR)

mi_db_server_console_echo.o : mi_db_server_console_echo.cpp mi_db_server_console_echo.h
	$(CC) -c -fPIC $(CFLAGS) mi_db_server_console_echo.cpp $(ADDITINAL_INCLUDE_DIR)

.PHONY : clean
clean:
	-rm *.o $(TARGET)

