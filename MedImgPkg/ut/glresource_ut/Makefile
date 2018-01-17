ver = release
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

LOCAL_LIB_DIR = /usr/local/lib

VPATH = $(INCLUDE_ROOT_DIR)

TARGET = $(BIN_DIR)/medimg_glresource_ut

ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -L$(LOCAL_LIB_DIR) -lmedimg_log -lmedimg_glresource -lGL -lGLEW -lboost_system -lboost_thread -lglut

$(TARGET):  mi_gl_resource_ut.cpp
	$(CC) $(CFLAGS) $(ADDITINAL_INCLUDE_DIR) -o $(TARGET)  mi_gl_resource_ut.cpp  $(ADDITIANL_DEPENDENCIES)

.PHONY : clean
clean:
	-rm $(TARGET)
