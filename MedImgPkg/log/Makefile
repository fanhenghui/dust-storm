ver = release
CC  = g++ -std=c++11 -Wall 

ifeq ($(ver) , release)
CFLAGS += -O3
else
CFLAGS += -g
endif

DEFS = -DBOOST_LOG_DYN_LINK
CFLAGS += $(DEFS)

ROOT = ..
INCLUDE_ROOT_DIR = $(ROOT)/include
INCLUDE_DIR = $(INCLUDE_ROOT_DIR)/log
BIN_DIR = $(ROOT)/bin

LOCAL_LIB_DIR = /usr/local/lib

VPATH = $(INCLUDE_ROOT_DIR) : $(INCLUDE_DIR)

OBJS = mi_logger.o mi_logger_util.o
TARGET = $(BIN_DIR)/libmedimg_log.so

EXPORT_HEADER = mi_log_export.h
ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR) -I$(INCLUDE_DIR)
ADDITIANL_DEPENDENCIES = -L$(LOCAL_LIB_DIR) -lpthread -lboost_system -lboost_filesystem -lboost_thread -lboost_log -lboost_log_setup

$(TARGET): $(OBJS)
	$(CC) -shared $(CFLAGS) -o $(TARGET) $(OBJS) $(ADDITIANL_DEPENDENCIES)

mi_logger_util.o: mi_logger_util.cpp mi_logger_util.h mi_logger.o
	$(CC) -c -fPIC $(CFLAGS) mi_logger_util.cpp $(ADDITINAL_INCLUDE_DIR)

mi_logger.o: mi_logger.cpp mi_logger.h
	$(CC) -c -fPIC $(CFLAGS) mi_logger.cpp $(ADDITINAL_INCLUDE_DIR)

.PHONY : clean
clean:
	-rm $(OBJS) $(TARGET)