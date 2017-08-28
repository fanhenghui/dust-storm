ver = debug
CC  = g++ -std=c++11 -Wall 

ifeq ($(ver) , release)
else
CFLAGS += -g
endif

ROOT = ..
INCLUDE_ROOT_DIR = $(ROOT)/include
INCLUDE_DIR = $(INCLUDE_ROOT_DIR)/util
BIN_DIR = $(ROOT)/bin

VPATH = $(INCLUDE_ROOT_DIR) : $(INCLUDE_DIR)

OBJS = mi_exception.o mi_configuration.o mi_model_interface.o mi_socket_client.o mi_ipc_client_proxy.o \
mi_file_util.o
TARGET = $(BIN_DIR)/libmedimg_util.so

EXPORT_HEADER = mi_util_export.h
COMMON_HEADER = mi_ipc_common.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) -I$(INCLUDE_DIR)
ADDITIANL_DEPENDENCIES = -lboost_system -lboost_filesystem -lboost_thread

$(TARGET): $(OBJS)
	$(CC) -shared $(CFLAGS) -o $(TARGET) $(OBJS) $(ADDITIANL_DEPENDENCIES)
$(OBJS):%.o:%.cpp %.h $(COMMON_HEADER)
	$(CC) -c -fPIC $(CFLAGS) $< -o $@ $(ADDITINAL_INCLUDE_DIR)

.PHONY : clean
clean:
	-rm $(OBJS) $(TARGET)