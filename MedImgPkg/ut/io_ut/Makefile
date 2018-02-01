ver = release
CC  = g++ -std=c++11 -Wall -Wno-unused-function

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

TARGET = $(BIN_DIR)/medimg_io_ut
OBJECTS = mi_pacs_ut.o mi_targa_ut.o mi_dicom_loader_ut.o mi_rt_ut.o mi_md5_ut.o mi_db_ut.o
MAINOBJ = mi_io_ut.o

ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -L$(LOCAL_LIB_DIR) -lpthread -lboost_system -lboost_thread -lboost_log -lboost_log_setup -ldcmdata -ldcmimage -ldcmjpeg -ldcmnet -loflog -lofstd \
-lmedimg_log -lmedimg_util -lmedimg_arithmetic -lmedimg_io 

$(TARGET):  $(OBJECTS) $(MAINOBJ)
	$(CC) $(CFLAGS) $(ADDITINAL_INCLUDE_DIR) -o $(TARGET) $(OBJECTS) $(MAINOBJ) $(ADDITIANL_DEPENDENCIES)

$(MAINOBJ): $(OBJECTS) mi_io_ut.cpp 
	$(CC) -c -fPIC $(CFLAGS) mi_io_ut.cpp $(ADDITINAL_INCLUDE_DIR)

$(OBJECTS): %.o: %.cpp
	$(CC) -c -fPIC $(CFLAGS) -o $@ $< $(ADDITINAL_INCLUDE_DIR)

.PHONY : clean
clean:
	-rm *.o
	-rm $(TARGET)
