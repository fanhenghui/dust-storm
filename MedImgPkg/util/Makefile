ver = release
CC  = g++ -std=c++11 -Wall 

ifeq ($(ver) , release)
CFLAGS += -O3
else
CFLAGS += -g
endif

DEF = -DBOOST_LOG_DYN_LINK
CFLAGS += $(DEF)

ROOT = ..
INCLUDE_ROOT_DIR = $(ROOT)/include
INCLUDE_DIR = $(INCLUDE_ROOT_DIR)/util
BIN_DIR = $(ROOT)/bin

LOCAL_LIB_DIR = /usr/local/lib

VPATH = $(INCLUDE_ROOT_DIR) : $(INCLUDE_DIR)

INDEPENDENT_OBJS = mi_model_interface.o mi_file_util.o mi_operation_factory.o mi_time_util.o mi_uid.o

SOCKET_OBJS = mi_socket_client.o mi_socket_server.o mi_ipc_client_proxy.o mi_ipc_server_proxy.o

TARGET = $(BIN_DIR)/libmedimg_util.so

EXPORT_HEADER = mi_util_export.h
COMMON_HEADER = mi_ipc_common.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) -I$(INCLUDE_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -L$(LOCAL_LIB_DIR) -lpthread -lboost_system -lboost_filesystem -lboost_thread -luuid \
-lboost_log -lmedimg_log

$(TARGET): $(INDEPENDENT_OBJS) $(SOCKET_OBJS)
	$(CC) -shared $(CFLAGS) -o $(TARGET) $(INDEPENDENT_OBJS) $(SOCKET_OBJS) $(ADDITIANL_DEPENDENCIES)

$(INDEPENDENT_OBJS):%.o:%.cpp %.h $(COMMON_HEADER)
	$(CC) -c -fPIC $(CFLAGS) -o $@ $< $(ADDITINAL_INCLUDE_DIR)

mi_ipc_server_proxy.o: mi_ipc_server_proxy.cpp mi_ipc_server_proxy.h mi_socket_server.o
	$(CC) -c -fPIC $(CFLAGS) mi_ipc_server_proxy.cpp $(ADDITINAL_INCLUDE_DIR)
mi_ipc_client_proxy.o: mi_ipc_client_proxy.cpp mi_ipc_client_proxy.h mi_socket_client.o
	$(CC) -c -fPIC $(CFLAGS) mi_ipc_client_proxy.cpp $(ADDITINAL_INCLUDE_DIR)

mi_socket_client.o: mi_socket_client.cpp mi_socket_client.h 
	$(CC) -c -fPIC $(CFLAGS) mi_socket_client.cpp $(ADDITINAL_INCLUDE_DIR)
mi_socket_server.o: mi_socket_server.cpp mi_socket_server.h 
	$(CC) -c -fPIC $(CFLAGS) mi_socket_server.cpp $(ADDITINAL_INCLUDE_DIR)
	

.PHONY : clean
clean:
	-rm *.o $(TARGET)