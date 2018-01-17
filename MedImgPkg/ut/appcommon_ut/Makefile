ver = release
CC  = g++ -std=c++11 -Wall -Wno-unused-function

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

LOCAL_LIB_DIR = /usr/local/lib

VPATH = $(INCLUDE_ROOT_DIR)

TARGET = $(BIN_DIR)/medimg_appcommon_ut

ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -L$(LOCAL_LIB_DIR) -lpthread -lboost_system -lboost_filesystem -lboost_thread -lboost_log -lboost_log_setup -lprotobuf \
-lmedimg_log -lmedimg_util -lmedimg_arithmetic -lmedimg_io -lmedimg_glresource -lmedimg_renderalgo -lmedimg_appcommon

$(TARGET):mi_init_ut.cpp mi_appcommon_ut.cpp 
	$(CC) $(CFLAGS) -o $(TARGET) mi_init_ut.cpp mi_appcommon_ut.cpp $(ADDITINAL_INCLUDE_DIR) $(ADDITIANL_DEPENDENCIES)

.PHONY : clean
clean:
	-rm *.o $(TARGET)
