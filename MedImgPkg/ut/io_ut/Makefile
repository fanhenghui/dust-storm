ver = debug
CC  = g++ -std=c++11 -Wall 

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

TARGET = $(BIN_DIR)/medimg_io_ut

ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -lmedimg_log -lmedimg_util -lmedimg_arithmetic -lmedimg_io -lboost_system -lboost_thread -ldcmdata -ldcmimage -ldcmjpeg -ldcmnet -loflog -lofstd

$(TARGET):  mi_io_ut.cpp mi_pacs_ut.cpp mi_targa_ut.cpp mi_dicom_loader_ut.cpp
	$(CC) $(CFLAGS) $(ADDITINAL_INCLUDE_DIR) -o $(TARGET) mi_targa_ut.cpp mi_pacs_ut.cpp mi_dicom_loader_ut.cpp mi_io_ut.cpp  $(ADDITIANL_DEPENDENCIES)

.PHONY : clean
clean:
	-rm $(TARGET)
