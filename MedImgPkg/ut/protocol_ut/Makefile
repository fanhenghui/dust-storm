ver = debug
CC  = g++ -std=c++11 -Wall 

ifeq ($(ver) , release)
CFLAGS += -O3
else
CFLAGS += -g
endif

ROOT = ../..
INCLUDE_ROOT_DIR = $(ROOT)/include
BIN_DIR = $(ROOT)/bin

VPATH = $(INCLUDE_ROOT_DIR) $(GOOGLE_DIR)

OBJS = mi_message.pb.o mi_protocol_ut.o
TARGET = $(BIN_DIR)/medimg_protocol_ut

ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -lprotobuf

all: pre-build main-build
pre-build:
	sh protoc-gen.sh
main-build: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(ADDITIANL_DEPENDENCIES) $(ADDITINAL_INCLUDE_DIR)

mi_protocol_ut.o: mi_protocol_ut.cpp  mi_message.pb.o
	$(CC) -c -fPIC $(CFLAGS) mi_protocol_ut.cpp $(ADDITINAL_INCLUDE_DIR)

mi_message.pb.o: mi_message.pb.h mi_message.pb.cc 
	$(CC) -c -fPIC $(CFLAGS) mi_message.pb.cc $(ADDITINAL_INCLUDE_DIR)

.PHONY : clean
clean:
	-rm $(TARGET) $(OBJS)
	-rm *.pb.cc *.pb.h
