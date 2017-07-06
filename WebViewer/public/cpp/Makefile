CC = g++ -std=c++11 -Wall
TARGET = be
$(TARGET) : be.cpp common.h img_gen.o 
	$(CC)  -o be be.cpp img_gen.o  -lpthread

img_gen.o : img_gen.h img_gen.cpp
	$(CC)  -c img_gen.cpp

.PHONY : clean
clean :
	rm $(TARGET) img_gen.o