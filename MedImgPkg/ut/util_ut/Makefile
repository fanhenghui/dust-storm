ver = debug
CC  = g++ -std=c++11 -Wall 

ifeq ($(ver) , release)

else
CFLAGS += -g
endif

ROOT = ../..
INCLUDE_ROOT_DIR = $(ROOT)/include
BIN_DIR = $(ROOT)/bin

VPATH = $(INCLUDE_ROOT_DIR)

TARGET = $(BIN_DIR)/medimg_util_ut

ADDITINAL_INCLUDE_DIR = -I$(INCLUDE_ROOT_DIR)
ADDITIANL_DEPENDENCIES = -L$(BIN_DIR) -lmedimg_util -lboost_system -lboost_filesystem -lboost_thread

$(TARGET):  mi_test_message_queue.o
	$(CC) $(CFLAGS) $(ADDITINAL_INCLUDE_DIR) -o $(TARGET)  mi_util_ut.cpp mi_test_message_queue.o $(ADDITIANL_DEPENDENCIES)

mi_test_message_queue.o : mi_test_message_queue.cpp
	$(CC) $(CFLAGS) $(ADDITINAL_INCLUDE_DIR) -c mi_test_message_queue.cpp
.PHONY : clean
clean:
	-rm $(TARGET) mi_test_message_queue.o