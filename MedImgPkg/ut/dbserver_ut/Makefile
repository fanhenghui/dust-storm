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

VPATH = $(INCLUDE_ROOT_DIR)

TARGET = $(BIN_DIR)/dbserver_ut

ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR)  -lpthread -lboost_system -lboost_filesystem -lboost_thread -lboost_log -lboost_log_setup -lprotobuf \
-lmedimg_log -lmedimg_util -lmedimg_arithmetic -lmedimg_io -lmedimg_glresource -lmedimg_renderalgo -lmedimg_appcommon

$(TARGET): mi_db_server_ut.o 
	$(CC) $(CFLAGS) -o $(TARGET) mi_db_server_ut.o $(OBJS) $(ADDITIANL_DEPENDENCIES)

mi_db_server_ut.o: mi_db_server_ut.cpp 
	$(CC) -c -fPIC $(CFLAGS) mi_db_server_ut.cpp $(ADDITINAL_INCLUDE_DIR)

.PHONY : clean
clean:
	-rm *.o $(TARGET)
