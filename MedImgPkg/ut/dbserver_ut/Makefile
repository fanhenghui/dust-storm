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

PROTOCO_LIB_DIR = /usr/local/lib
GOOGLE_INCLUDE_DIR = /usr/local/include/google/
PROTOCO_INCLUDE_DIR = /usr/local/include/google/protobuf

VPATH = $(INCLUDE_ROOT_DIR)

TARGET = $(BIN_DIR)/dbserver_ut

ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) -I$(GOOGLE_INCLUDE_DIR) -I$(PROTOCO_INCLUDE_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -L$(PROTOCO_LIB_DIR) -lpthread -lboost_system -lboost_filesystem -lboost_thread -lboost_log -lboost_log_setup -lGL -lGLEW -lprotobuf \
-lmedimg_log -lmedimg_util

all: pre-build main-build

pre-build:
	sh protoc-gen.sh

main-build: $(TARGET)

$(TARGET): mi_message.pb.o mi_db_server_ut.o 
	$(CC) $(CFLAGS) -o $(TARGET) mi_message.pb.o mi_db_server_ut.o $(OBJS) $(ADDITIANL_DEPENDENCIES)

mi_db_server_ut.o: mi_db_server_ut.cpp 
	$(CC) -c -fPIC $(CFLAGS) mi_db_server_ut.cpp $(ADDITINAL_INCLUDE_DIR)

mi_message.pb.o: mi_message.pb.h mi_message.pb.cc 
	$(CC) -c -fPIC $(CFLAGS) mi_message.pb.cc $(ADDITINAL_INCLUDE_DIR)

.PHONY : clean
clean:
	-rm *.o $(TARGET)
	-rm *.pb.cc *.pb.h

