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

LOCAL_LIB_DIR = /usr/local/lib

VPATH = $(INCLUDE_ROOT_DIR)

TARGET = $(BIN_DIR)/dcmdbimporter

ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -L$(LOCAL_LIB_DIR) -lboost_system -lboost_filesystem -lmedimg_util -lmedimg_io -lmedimg_appcommon

$(TARGET): mi_dcm_db_importer.cpp
	$(CC) $(CFLAGS) -o $(TARGET) mi_dcm_db_importer.cpp $(ADDITINAL_INCLUDE_DIR) $(ADDITIANL_DEPENDENCIES)

.PHONY : clean
clean:
	-rm $(TARGET)