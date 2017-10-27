ver = debug
CC = g++
CFLAGS += -std=c++11 -Wall

ifeq ($(ver) , release)	
CFLAGS += -O3
else
CFLAGS += -g
endif

DEFS = -DBOOST_LOG_DYN_LINK
CFLAGS += $(DEFS)

ROOT = ../..
INCLUDE_ROOT_DIR = $(ROOT)/include
BIN_DIR = $(ROOT)/bin

VPATH = $(INCLUDE_ROOT_DIR)

TARGET = $(BIN_DIR)/majorextract

ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -lboost_system -lboost_filesystem -lboost_thread -lboost_log -lboost_log_setup -lmedimg_util -lmedimg_arithmetic -lmedimg_io

$(TARGET): major_extract.cpp
	$(CC) $(CFLAGS) -o $(TARGET) major_extract.cpp $(ADDITINAL_INCLUDE_DIR) $(ADDITIANL_DEPENDENCIES)

.PHONY : clean
clean:
	-rm $(TARGET)