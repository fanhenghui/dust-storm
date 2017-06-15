INCLUDE = ../include/
CC = g++
OBJS = class0.o class1.o
TARGET = ../lib/libclasses.a
VPATH = ../include
vpath %.h ../include/

$(TARGET): $(OBJS)
	ar -rcs $(TARGET) $(OBJS)
#	-rm $(OBJS)
#	mv $(TARGET) ../lib 

class0.o : class0.cpp class0.h
	$(CC) -c class0.cpp -I$(INCLUDE)

class1.o : class1.cpp class1.h
	$(CC) -c class1.cpp -I$(INCLUDE)

.PHONY : clean
clean :
	-rm $(TARGET) $(OBJS)
