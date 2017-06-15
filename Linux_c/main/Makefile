ver = debug

EXEDIR = ../bin
LIB_DIR = ../lib
INCLUDE_DIR = ../include

VPATH = $(INCLUDE_DIR)
vpath %.h $(INCLUDE_DIR)

ifeq ($(ver),debug)
CFLAGS += -g
endif

CC = g++
TARGET = $(EXEDIR)/make_file_test

$(TARGET) : linux_c_test.cpp ../lib/libclasses.a
	$(CC) $(CFLAGS) linux_c_test.cpp -I$(INCLUDE_DIR) -L$(LIB_DIR) -lclasses -lboost_system -lboost_thread  -o $(TARGET)

.PHONY : clean
clean :
	-rm $(TARGET)