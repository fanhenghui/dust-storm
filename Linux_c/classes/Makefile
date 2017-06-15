ver = debug
lib = shared

BIN_DIR = ../bin
LIB_DIR = ../lib
INCLUDE_DIR = ../include

VPATH = $(INCLUDE_DIR)
vpath %.h $(INCLUDE_DIR)

ifeq ($(ver),debug)
CFLAGS += -c -g -fPIC
else
CFLAGS += -c -fPIC
endif

DEFS = -D TEST
CFLAGS += $(DEFS)

CC = g++

OBJS = class0.o class1.o
TARGET_NAME = libclasses

ifeq ($(lib),shared)
TARGET = $(BIN_DIR)/$(TARGET_NAME).so
$(TARGET) : $(OBJS)
	$(CC) -shared -fPIC -o $(TARGET) $(OBJS)
else
TARGET = $(LIB_DIR)/$(TARGET_NAME).a
$(TARGET): $(OBJS)
	ar -rcs $(TARGET) $(OBJS)
endif

class0.o : class0.cpp class0.h
	$(CC) $(CFLAGS) class0.cpp -I$(INCLUDE_DIR)

class1.o : class1.cpp class1.h
	$(CC) $(CFLAGS) class1.cpp -I$(INCLUDE_DIR)

.PHONY : clean
clean :
	-rm $(TARGET) $(OBJS)
