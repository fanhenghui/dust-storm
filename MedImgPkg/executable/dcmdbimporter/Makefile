ver = release
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

OBJS = mi_dcm_db_importer.o mi_dcm_file_browser.o
TARGET = $(BIN_DIR)/dcmdbimporter

ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -L$(LOCAL_LIB_DIR) -lpthread -lboost_system -lboost_thread -lboost_log -lboost_log_setup -lprotobuf \
-lmedimg_log -lmedimg_util -lmedimg_io

$(TARGET): mi_dcm_db_importer.o mi_dcm_file_browser.o
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(ADDITIANL_DEPENDENCIES)

mi_dcm_db_importer.o: mi_dcm_db_importer.cpp mi_dcm_file_browser.o
	$(CC) -c -fPIC $(CFLAGS) mi_dcm_db_importer.cpp $(ADDITINAL_INCLUDE_DIR) 

mi_dcm_file_browser.o: mi_dcm_file_browser.cpp mi_dcm_file_browser.h
	$(CC) -c -fPIC $(CFLAGS) mi_dcm_file_browser.cpp $(ADDITINAL_INCLUDE_DIR) 

.PHONY : clean
clean:
	-rm *.o
	-rm $(TARGET)