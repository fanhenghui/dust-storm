VPATH = ../include/

CC = g++
TARGET = ../bin/make_file_test
INCLUDE = ../include
LIB_PATH = ../lib

$(TARGET) : linux_c_test.cpp ../lib/libclasses.a
	$(CC) linux_c_test.cpp -I$(INCLUDE) -L$(LIB_PATH) -lclasses -lboost_system -lboost_thread  -o $(TARGET)

.PHONY : clean
clean :
	-rm $(TARGET)