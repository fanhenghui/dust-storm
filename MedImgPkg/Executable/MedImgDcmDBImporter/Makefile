ver = debug
CC = g++
CFLAGS += -std=c++11 -Wall

ifeq ($(ver) , release)	
else
CFLAGS += -g
endif

ROOT = ../..
INCLUDE_ROOT_DIR = $(ROOT)/Include
BIN_DIR = $(ROOT)/bin

VPATH = $(INCLUDE_ROOT_DIR)

TARGET = $(BIN_DIR)/dcmdbimporter

ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -lboost_system -lboost_filesystem -lmedimg_util -lmedimg_io -lmedimg_appcommon

$(TARGET): mi_dcm_db_importer.cpp
	$(CC) $(CFLAGS) -o $(TARGET) mi_dcm_db_importer.cpp $(ADDITINAL_INCLUDE_DIR) $(ADDITIANL_DEPENDENCIES)

.PHONY : clean
clean:
	-rm $(TARGET)