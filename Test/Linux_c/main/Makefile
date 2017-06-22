ver = release
lib = shared

BIN_DIR = ../bin
LIB_DIR = ../lib
INCLUDE_DIR = ../include

VPATH = $(INCLUDE_DIR)
vpath %.h $(INCLUDE_DIR)

ifeq ($(ver),debug)
CFLAGS += -g
endif

CC = g++
TARGET_NAME = main
TARGET = $(BIN_DIR)/$(TARGET_NAME)

ifeq ($(lib),shared) 
$(TARGET) : linux_c_test.cpp
#	$(CC) $(CFLAGS) linux_c_test.cpp -I$(INCLUDE_DIR) -L$(BIN_DIR) -lclasses -lboost_system -lboost_thread -o $(TARGET)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -L$(BIN_DIR) -o $(TARGET) linux_c_test.cpp  ../bin/libclasses.so 
else
$(TARGET) : linux_c_test.cpp $(LIB_DIR)/libclasses.a
	$(CC) linux_c_test.cpp -I$(INCLUDE_DIR) -L$(LIB_DIR) -lclasses -lboost_system -lboost_thread  -o $(TARGET)
#	$(CC) $(CFLAGS) -static linux_c_test.cpp -I$(INCLUDE_DIR) -L$(LIB_DIR) -lclasses  -o $(TARGET)
endif

.PHONY : clean
clean :
	-rm $(TARGET)