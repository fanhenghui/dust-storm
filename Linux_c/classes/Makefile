ver = debug

EXEDIR = ../bin
LIB_DIR = ../lib
INCLUDE_DIR = ../include

VPATH = $(INCLUDE_DIR)
vpath %.h $(INCLUDE_DIR)

ifeq ($(ver),debug)
CFLAGS += -c -g
else
CFLAGS += -c
endif

DEFS = -D TEST
CFLAGS += $(DEFS)

CC = g++

OBJS = class0.o class1.o
TARGET = $(LIB_DIR)/libclasses.a

$(TARGET): $(OBJS)
	ar -rcs $(TARGET) $(OBJS)
#	-rm $(OBJS)
#	mv $(TARGET) ../lib 

class0.o : class0.cpp class0.h
	$(CC) $(CFLAGS) class0.cpp -I$(INCLUDE_DIR)

class1.o : class1.cpp class1.h
	$(CC) $(CFLAGS) class1.cpp -I$(INCLUDE_DIR)

.PHONY : clean
clean :
	-rm $(TARGET) $(OBJS)
