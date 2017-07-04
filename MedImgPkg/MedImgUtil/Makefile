ver = release
CC  = g++ -std=c++11 -Wall 

ifeq ($(ver) , release)
CFLAGS += -c -fPIC
else
CFLAGS += -c -g -fPIC
endif

ROOT = ..
INCLUDE_ROOT_DIR = $(ROOT)/Include
INCLUDE_DIR = $(INCLUDE_ROOT_DIR)/MedImgUtil
BIN_DIR = $(ROOT)/bin

VPATH = $(INCLUDE_ROOT_DIR) : $(INCLUDE_DIR)

OBJS = mi_exception.o mi_configuration.o mi_model_interface.o mi_socket_client.o mi_socket_server.o mi_ipc_client_proxy.o
TARGET = $(BIN_DIR)/libmedimg_util.so

EXPORT_HEADER = mi_util_export.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) -I$(INCLUDE_DIR)
ADDITIANL_DEPENDENCIES = -lboost_system -lboost_filesystem -lboost_thread

$(TARGET): $(OBJS)
	$(CC) -shared -o $(TARGET) $(OBJS) $(ADDITIANL_DEPENDENCIES)

mi_ipc_client_proxy.o: $(EXPORT_HEADER) mi_ipc_common.h mi_ipc_client_proxy.h mi_ipc_client_proxy.cpp mi_socket_client.o mi_socket_server.o
	$(CC) $(CFLAGS) mi_ipc_client_proxy.cpp $(ADDITINAL_INCLUDE_DIR)

mi_socket_client.o: $(EXPORT_HEADER) mi_ipc_common.h mi_socket_client.h mi_socket_client.cpp mi_exception.o
	$(CC) $(CFLAGS) mi_socket_client.cpp $(ADDITINAL_INCLUDE_DIR)

mi_socket_server.o: $(EXPORT_HEADER) mi_ipc_common.h mi_socket_server.h mi_socket_server.cpp
	$(CC) $(CFLAGS) mi_socket_server.cpp $(ADDITINAL_INCLUDE_DIR)

mi_exception.o: $(EXPORT_HEADER) mi_exception.h mi_exception.cpp
	$(CC) $(CFLAGS) mi_exception.cpp $(ADDITINAL_INCLUDE_DIR)

mi_configuration.o: $(EXPORT_HEADER) mi_configuration.h mi_configuration.cpp
	$(CC) $(CFLAGS) mi_configuration.cpp $(ADDITINAL_INCLUDE_DIR)

mi_model_interface.o: $(EXPORT_HEADER) mi_model_interface.h mi_model_interface.cpp
	$(CC) $(CFLAGS) mi_model_interface.cpp $(ADDITINAL_INCLUDE_DIR)

.PHONY : clean
clean:
	-rm $(OBJS) $(TARGET)
